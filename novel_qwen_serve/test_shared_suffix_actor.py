from __future__ import annotations

import unittest

import numpy as np

from novel_qwen_serve.shared_suffix_actor import SharedSuffixCorpus


class SharedSuffixCorpusTests(unittest.TestCase):
    def test_delta_cursor_pages_without_skipping_backlog(self) -> None:
        corpus = SharedSuffixCorpus(
            max_sequences=100,
            max_total_tokens=10_000,
            max_sequence_tokens=32,
            use_object_store=False,
        )
        for i in range(30):
            corpus.add_sequence(f"replica-{i % 5}", str(i), np.arange(i, i + 8))

        cursor = 0
        seen_versions: list[int] = []
        while cursor < corpus.stats()["version"]:
            cursor, entries = corpus.delta_since(cursor, limit=3, skip_replica="replica-0")
            seen_versions.extend(entry["version"] for entry in entries)

        self.assertGreater(len(seen_versions), 3)
        self.assertEqual(seen_versions, [v for v in range(1, 31) if (v - 1) % 5 != 0])

    def test_delta_uses_version_index_after_large_cursor(self) -> None:
        corpus = SharedSuffixCorpus(
            max_sequences=2_000,
            max_total_tokens=1_000_000,
            max_sequence_tokens=16,
            use_object_store=False,
        )
        for i in range(2_000):
            corpus.add_sequence(f"replica-{i % 5}", str(i), np.arange(i, i + 8))

        cursor, entries = corpus.delta_since(1_990, limit=100, skip_replica="replica-4")
        self.assertEqual(cursor, 2_000)
        self.assertEqual([entry["version"] for entry in entries], [1991, 1992, 1993, 1994, 1996, 1997, 1998, 1999])

    def test_eviction_keeps_log_bounded_to_live_entries(self) -> None:
        corpus = SharedSuffixCorpus(
            max_sequences=5,
            max_total_tokens=10_000,
            max_sequence_tokens=16,
            use_object_store=False,
        )
        for i in range(20):
            corpus.add_sequence("replica-0", str(i), np.arange(i, i + 8))

        stats = corpus.stats()
        self.assertEqual(stats["sequences"], 5)
        self.assertEqual(stats["version_log_entries"], 5)
        self.assertEqual(stats["log_base_version"], 16)

    def test_duplicate_hits_do_not_pin_old_log_entries(self) -> None:
        corpus = SharedSuffixCorpus(
            max_sequences=5,
            max_total_tokens=10_000,
            max_sequence_tokens=16,
            use_object_store=False,
        )
        duplicate = np.arange(8)
        corpus.add_sequence("replica-0", "first", duplicate)
        for i in range(10):
            corpus.add_sequence("replica-0", f"dup-{i}", duplicate)
        for i in range(20):
            corpus.add_sequence("replica-1", str(i), np.arange(i + 10, i + 18))

        stats = corpus.stats()
        self.assertEqual(stats["sequences"], 5)
        self.assertEqual(stats["version_log_entries"], 5)


if __name__ == "__main__":
    unittest.main()
