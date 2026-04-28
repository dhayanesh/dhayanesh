from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from typing import Any

import numpy as np

from novel_qwen_serve.shared_suffix_actor import SharedSuffixCorpus


def _token_batch(entries: int, tokens: int) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [
        rng.integers(0, 32_000, size=tokens, dtype=np.int32)
        for _ in range(entries)
    ]


def _print_profile(profile: cProfile.Profile) -> None:
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(12)
    print(stream.getvalue())


def run_direct(args: argparse.Namespace) -> None:
    corpus = SharedSuffixCorpus(
        max_sequences=args.entries,
        max_total_tokens=args.entries * args.tokens * 2,
        max_sequence_tokens=args.tokens,
        use_object_store=False,
    )
    for i, token_ids in enumerate(_token_batch(args.entries, args.tokens)):
        corpus.add_sequence(f"replica-{i % args.replicas}", str(i), token_ids)

    cursors = [0 for _ in range(args.replicas)]

    def sync_rounds() -> int:
        hydrated = 0
        for i in range(args.rounds):
            replica_idx = i % args.replicas
            cursor, batch = corpus.delta_since(
                cursors[replica_idx],
                args.batch,
                skip_replica=f"replica-{replica_idx}",
            )
            cursors[replica_idx] = cursor
            hydrated += len(batch)
        return hydrated

    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    hydrated = sync_rounds()
    profile.disable()
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(
        "direct "
        f"replicas={args.replicas} entries={args.entries} hydrated={hydrated} "
        f"elapsed_ms={elapsed_ms:.2f} stats={corpus.stats()}"
    )
    _print_profile(profile)


def run_ray(args: argparse.Namespace) -> None:
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=max(args.replicas + 1, 2), include_dashboard=False)

    remote_cls = ray.remote(num_cpus=1, max_concurrency=64)(SharedSuffixCorpus)
    corpus = remote_cls.remote(
        max_sequences=args.entries,
        max_total_tokens=args.entries * args.tokens * 2,
        max_sequence_tokens=args.tokens,
        use_object_store=True,
    )
    add_refs = [
        corpus.add_sequence.remote(f"replica-{i % args.replicas}", str(i), token_ids)
        for i, token_ids in enumerate(_token_batch(args.entries, args.tokens))
    ]
    ray.get(add_refs)

    cursors = [0 for _ in range(args.replicas)]
    hydrated = 0
    started = time.perf_counter()
    for i in range(args.rounds):
        replica_idx = i % args.replicas
        cursor, batch = ray.get(
            corpus.delta_since.remote(
                cursors[replica_idx],
                args.batch,
                f"replica-{replica_idx}",
            )
        )
        refs: list[Any] = [entry["token_ref"] for entry in batch if "token_ref" in entry]
        if refs:
            ray.get(refs)
        cursors[replica_idx] = cursor
        hydrated += len(batch)
    elapsed_ms = (time.perf_counter() - started) * 1000
    stats = ray.get(corpus.stats.remote())
    print(
        "ray "
        f"replicas={args.replicas} entries={args.entries} hydrated={hydrated} "
        f"elapsed_ms={elapsed_ms:.2f} stats={stats}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "ray"), default="direct")
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--entries", type=int, default=20_000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicas < 5:
        raise SystemExit("--replicas must be at least 5 for this benchmark")
    if args.mode == "ray":
        run_ray(args)
    else:
        run_direct(args)


if __name__ == "__main__":
    main()
