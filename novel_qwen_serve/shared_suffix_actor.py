from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CorpusEntry:
    version: int
    key: str
    source_replica: str
    source_request_id: str
    token_ids: np.ndarray
    created_at: float
    token_ref: Any | None = None

    def as_dict(self, *, use_object_refs: bool) -> dict[str, Any]:
        payload = {
            "version": self.version,
            "key": self.key,
            "source_replica": self.source_replica,
            "source_request_id": self.source_request_id,
            "created_at": self.created_at,
        }
        if use_object_refs and self.token_ref is not None:
            payload["token_ref"] = self.token_ref
            payload["num_tokens"] = int(self.token_ids.size)
        else:
            payload["token_ids"] = self.token_ids
        return payload


class SharedSuffixCorpus:
    """Ray actor body for cross-replica suffix history.

    The actor stores completed prompt-tail + response token sequences. Replicas
    periodically copy new entries into their local Arctic suffix tree. Keeping
    speculation local avoids a remote actor call on every vLLM decode step.
    """

    def __init__(
        self,
        max_sequences: int = 75_000,
        max_total_tokens: int = 12_000_000,
        max_sequence_tokens: int = 4096,
        use_object_store: bool = True,
    ) -> None:
        self.max_sequences = max_sequences
        self.max_total_tokens = max_total_tokens
        self.max_sequence_tokens = max_sequence_tokens
        self.use_object_store = use_object_store
        self._entries: OrderedDict[str, CorpusEntry] = OrderedDict()
        self._version_log: list[CorpusEntry] = []
        self._log_base_version = 1
        self._version = 0
        self._total_tokens = 0
        self._dedup_hits = 0
        self._evictions = 0
        self._ray = None
        if use_object_store:
            try:
                import ray

                if ray.is_initialized():
                    self._ray = ray
            except Exception:
                self._ray = None

    def add_sequence(
        self,
        source_replica: str,
        source_request_id: str,
        token_ids: list[int] | np.ndarray,
    ) -> int:
        arr = np.asarray(token_ids, dtype=np.int32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            return self._version
        if arr.size > self.max_sequence_tokens:
            arr = arr[-self.max_sequence_tokens :]
        arr = np.ascontiguousarray(arr, dtype=np.int32)

        key = hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()
        if key in self._entries:
            self._dedup_hits += 1
            return self._version

        self._version += 1
        token_ref = self._ray.put(arr) if self._ray is not None else None
        entry = CorpusEntry(
            version=self._version,
            key=key,
            source_replica=source_replica,
            source_request_id=source_request_id,
            token_ids=arr,
            created_at=time.time(),
            token_ref=token_ref,
        )
        self._entries[key] = entry
        self._version_log.append(entry)
        self._total_tokens += int(arr.size)
        self._evict()
        return self._version

    def delta_since(
        self,
        version: int,
        limit: int = 256,
        skip_replica: str | None = None,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return a bounded page of deltas and the next cursor version.

        The first implementation returned the actor's latest version even when
        the page was capped by ``limit``. Replicas then skipped the rest of the
        backlog. This cursor advances only through entries actually scanned, so
        large bursts hydrate over several cheap pages.
        """

        limit = max(int(limit), 0)
        if limit == 0:
            return min(max(int(version), 0), self._version), []

        entries: list[dict[str, Any]] = []
        cursor_version = min(max(int(version), 0), self._version)
        if not self._version_log:
            return cursor_version, entries

        start_index = max(0, cursor_version - self._log_base_version + 1)
        for entry in self._version_log[start_index:]:
            cursor_version = entry.version
            if entry.key not in self._entries:
                continue
            if skip_replica is not None and entry.source_replica == skip_replica:
                continue
            entries.append(entry.as_dict(use_object_refs=self._ray is not None))
            if len(entries) >= limit:
                break
        else:
            cursor_version = self._version
        return cursor_version, entries

    def stats(self) -> dict[str, Any]:
        return {
            "version": self._version,
            "sequences": len(self._entries),
            "total_tokens": self._total_tokens,
            "version_log_entries": len(self._version_log),
            "log_base_version": self._log_base_version,
            "max_sequences": self.max_sequences,
            "max_total_tokens": self.max_total_tokens,
            "max_sequence_tokens": self.max_sequence_tokens,
            "object_store_enabled": self._ray is not None,
            "dedup_hits": self._dedup_hits,
            "evictions": self._evictions,
        }

    def _evict(self) -> None:
        while self._entries and (
            len(self._entries) > self.max_sequences
            or self._total_tokens > self.max_total_tokens
        ):
            _, entry = self._entries.popitem(last=False)
            self._total_tokens -= int(entry.token_ids.size)
            self._evictions += 1
        self._compact_version_log()

    def _compact_version_log(self) -> None:
        drop = 0
        for entry in self._version_log:
            if entry.key in self._entries:
                break
            drop += 1
        if drop:
            self._version_log = self._version_log[drop:]
            self._log_base_version = (
                self._version_log[0].version if self._version_log else self._version + 1
            )


def get_or_create_shared_suffix_actor(config: Any) -> Any:
    import ray

    remote_cls = ray.remote(num_cpus=1, max_concurrency=64)(SharedSuffixCorpus)
    try:
        return ray.get_actor(config.shared_actor_name)
    except ValueError:
        return remote_cls.options(
            name=config.shared_actor_name,
            lifetime="detached",
            namespace="serve",
        ).remote(
            max_sequences=config.shared_max_sequences,
            max_total_tokens=config.shared_max_total_tokens,
            max_sequence_tokens=config.shared_max_sequence_tokens,
            use_object_store=getattr(config, "shared_use_object_store", True),
        )
