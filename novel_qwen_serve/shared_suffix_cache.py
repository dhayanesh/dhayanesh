from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import defaultdict
from typing import Any, ClassVar, Hashable, Sequence

import numpy as np

from arctic_inference.suffix_decoding.cache import SuffixDecodingCache as _ArcticSuffixCache

logger = logging.getLogger(__name__)


def _as_int_list(token_ids: np.ndarray | Sequence[int], max_tokens: int | None = None) -> list[int]:
    if isinstance(token_ids, np.ndarray):
        arr = np.asarray(token_ids, dtype=np.int32).reshape(-1)
        if max_tokens is not None and arr.size > max_tokens:
            arr = arr[-max_tokens:]
        return [int(x) for x in arr.tolist()]
    values = [int(x) for x in token_ids]
    if max_tokens is not None and len(values) > max_tokens:
        return values[-max_tokens:]
    return values


class SharedBackedSuffixDecodingCache(_ArcticSuffixCache):
    """Arctic suffix cache with async cross-replica hydration.

    The inherited Arctic cache still performs the hot-path suffix matching in
    process. This subclass publishes completed request token sequences to a Ray
    actor and periodically hydrates remote sequences into its local global tree.
    """

    _shared_actor: ClassVar[Any | None] = None
    _replica_id: ClassVar[str] = "unknown-replica"
    _prompt_suffix_tokens: ClassVar[int] = 512
    _min_response_tokens: ClassVar[int] = 8
    _sync_interval_s: ClassVar[float] = 0.75
    _sync_batch: ClassVar[int] = 256
    _sync_timeout_s: ClassVar[float] = 0.10
    _instances: ClassVar[weakref.WeakSet["SharedBackedSuffixDecodingCache"]] = weakref.WeakSet()

    @classmethod
    def configure(
        cls,
        *,
        shared_actor: Any | None,
        replica_id: str,
        prompt_suffix_tokens: int,
        min_response_tokens: int,
        sync_interval_s: float,
        sync_batch: int,
        sync_timeout_s: float,
    ) -> None:
        cls._shared_actor = shared_actor
        cls._replica_id = replica_id
        cls._prompt_suffix_tokens = prompt_suffix_tokens
        cls._min_response_tokens = min_response_tokens
        cls._sync_interval_s = sync_interval_s
        cls._sync_batch = sync_batch
        cls._sync_timeout_s = sync_timeout_s

    def __init__(self, max_tree_depth: int = 64, max_cached_requests: int = -1):
        super().__init__(max_tree_depth=max_tree_depth, max_cached_requests=max_cached_requests)
        self._lock = threading.RLock()
        self._active_prompt_tokens: dict[Hashable, list[int]] = {}
        self._active_response_tokens: defaultdict[Hashable, list[int]] = defaultdict(list)
        self._shared_version = 0
        self._shared_hydrated = 0
        self._shared_publish_attempts = 0
        self._shared_publish_errors = 0
        self._shared_sync_errors = 0
        self._shared_hydrate_ms = 0.0
        self._shared_stop = threading.Event()
        self._shared_thread: threading.Thread | None = None
        self._instances.add(self)
        if self._shared_actor is not None and self._sync_interval_s > 0:
            self._shared_thread = threading.Thread(
                target=self._sync_loop,
                name=f"suffix-sync-{self._replica_id}",
                daemon=True,
            )
            self._shared_thread.start()

    @property
    def shared_stats(self) -> dict[str, Any]:
        return {
            "replica_id": self._replica_id,
            "shared_version": self._shared_version,
            "hydrated_sequences": self._shared_hydrated,
            "publish_attempts": self._shared_publish_attempts,
            "publish_errors": self._shared_publish_errors,
            "sync_errors": self._shared_sync_errors,
            "hydrate_ms": round(self._shared_hydrate_ms, 3),
            "sync_interval_s": self._sync_interval_s,
        }

    def start_request(
        self,
        req_id: Hashable,
        prompt_token_ids: np.ndarray | Sequence[int],
    ):
        with self._lock:
            self._active_prompt_tokens[req_id] = _as_int_list(
                prompt_token_ids, self._prompt_suffix_tokens
            )
            self._active_response_tokens.pop(req_id, None)
            return super().start_request(req_id, prompt_token_ids)

    def add_active_response(
        self,
        req_id: Hashable,
        token_ids: np.ndarray | Sequence[int],
    ):
        with self._lock:
            self._active_response_tokens[req_id].extend(_as_int_list(token_ids))
            return super().add_active_response(req_id, token_ids)

    def stop_request(self, req_id: Hashable):
        with self._lock:
            prompt_tokens = self._active_prompt_tokens.pop(req_id, [])
            response_tokens = self._active_response_tokens.pop(req_id, [])
            result = super().stop_request(req_id)
        self._publish_completed(req_id, prompt_tokens, response_tokens)
        return result

    def evict_cached_response(self, req_id: Hashable):
        with self._lock:
            return super().evict_cached_response(req_id)

    def speculate(
        self,
        req_id: Hashable,
        context: np.ndarray | Sequence[int],
        max_spec_tokens: int | None = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ):
        with self._lock:
            return super().speculate(
                req_id,
                context,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=max_spec_factor,
                max_spec_offset=max_spec_offset,
                min_token_prob=min_token_prob,
                use_tree_spec=use_tree_spec,
            )

    def _publish_completed(
        self,
        req_id: Hashable,
        prompt_tokens: list[int],
        response_tokens: list[int],
    ) -> None:
        if self._shared_actor is None:
            return
        if len(response_tokens) < self._min_response_tokens:
            return
        token_ids = prompt_tokens + response_tokens
        if not token_ids:
            return
        self._shared_publish_attempts += 1
        try:
            self._shared_actor.add_sequence.remote(
                self._replica_id,
                str(req_id),
                np.asarray(token_ids, dtype=np.int32),
            )
        except Exception:
            self._shared_publish_errors += 1
            logger.exception("Failed to publish suffix sequence to shared actor")

    def _sync_loop(self) -> None:
        while not self._shared_stop.wait(self._sync_interval_s):
            self.sync_shared_once()

    def sync_shared_once(self) -> dict[str, Any]:
        if self._shared_actor is None:
            return self.shared_stats
        try:
            import ray

            ref = self._shared_actor.delta_since.remote(
                self._shared_version,
                self._sync_batch,
                self._replica_id,
            )
            cursor_version, entries = ray.get(ref, timeout=self._sync_timeout_s)
            if entries:
                started = time.perf_counter()
                self._hydrate_entries(entries, ray=ray)
                self._shared_hydrate_ms += (time.perf_counter() - started) * 1000
            if cursor_version > self._shared_version:
                self._shared_version = int(cursor_version)
        except Exception:
            self._shared_sync_errors += 1
            logger.debug("Suffix cache shared sync failed", exc_info=True)
        return self.shared_stats

    def _hydrate_entries(self, entries: list[dict[str, Any]], *, ray: Any | None = None) -> None:
        materialized: list[tuple[dict[str, Any], np.ndarray]] = []
        refs: list[Any] = []
        ref_entries: list[dict[str, Any]] = []
        for entry in entries:
            if "token_ref" in entry:
                refs.append(entry["token_ref"])
                ref_entries.append(entry)
            else:
                materialized.append((entry, np.asarray(entry["token_ids"], dtype=np.int32)))

        if refs:
            if ray is None:
                import ray as ray_mod

                ray = ray_mod
            arrays = ray.get(refs, timeout=self._sync_timeout_s)
            materialized.extend(
                (entry, np.asarray(token_ids, dtype=np.int32))
                for entry, token_ids in zip(ref_entries, arrays, strict=True)
            )

        for entry, token_ids in materialized:
            token_ids = token_ids.reshape(-1)
            if token_ids.size == 0:
                continue
            req_id = f"shared:{entry['version']}:{entry['key']}"
            with self._lock:
                if req_id in self.cached_requests:
                    continue
                if req_id in self.active_requests:
                    continue
                super().start_request(req_id, np.empty(0, dtype=np.int32))
                super().add_active_response(req_id, np.ascontiguousarray(token_ids))
                super().stop_request(req_id)
                self._shared_hydrated += 1

    def __del__(self) -> None:
        try:
            self._shared_stop.set()
        except Exception:
            pass


def install_shared_suffix_cache(
    *,
    shared_actor: Any | None,
    replica_id: str,
    prompt_suffix_tokens: int,
    min_response_tokens: int,
    sync_interval_s: float,
    sync_batch: int,
    sync_timeout_s: float,
) -> None:
    """Patch Arctic's cache class before vLLM builds the suffix proposer."""

    SharedBackedSuffixDecodingCache.configure(
        shared_actor=shared_actor,
        replica_id=replica_id,
        prompt_suffix_tokens=prompt_suffix_tokens,
        min_response_tokens=min_response_tokens,
        sync_interval_s=sync_interval_s,
        sync_batch=sync_batch,
        sync_timeout_s=sync_timeout_s,
    )

    import arctic_inference.suffix_decoding as suffix_pkg
    import arctic_inference.suffix_decoding.cache as cache_mod

    suffix_pkg.SuffixDecodingCache = SharedBackedSuffixDecodingCache
    cache_mod.SuffixDecodingCache = SharedBackedSuffixDecodingCache


def get_local_shared_cache_stats() -> list[dict[str, Any]]:
    return [instance.shared_stats for instance in list(SharedBackedSuffixDecodingCache._instances)]
