#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""A/B direct-read throughput: 2 MiB-aligned versus 4 KiB-aligned file offsets.

  io_align.py FILE THREADS READS SEED

Reads READS random ranges with O_DIRECT (no page cache) using THREADS
concurrent preadv calls, for each arm in an interleaved order. Offsets are
drawn once per repetition and shifted per arm, so every arm reads the same
neighbourhoods. Read-only. Prints one JSON line per arm.
"""
import concurrent.futures
import json
import mmap
import os
import random
import sys
import time

MiB = 1 << 20
ARMS = [  # (name, extra offset from a 2 MiB boundary, length)
    ("aligned_2m_len_2m", 0, 2 * MiB),
    ("aligned_4k_len_2m", 4096 * 37, 2 * MiB),
    ("aligned_2m_len_closure", 0, 1_974_272),   # Qwen3.8 expert closure, 4 KiB-rounded
    ("aligned_4k_len_closure", 4096 * 37, 1_974_272),
]


def run(path, threads, reads, seed):
    size = os.path.getsize(path)
    rng = random.Random(seed)
    slots = (size - 8 * MiB) // (2 * MiB)
    bases = [rng.randrange(slots) * 2 * MiB for _ in range(reads)]
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    bufs = [mmap.mmap(-1, 4 * MiB) for _ in range(threads)]
    order = ARMS[:] if seed % 2 == 0 else ARMS[::-1]
    try:
        for name, shift, length in order:
            def one(i):
                buf = bufs[i % threads]
                got = os.preadv(fd, [memoryview(buf)[:length]], bases[i] + shift)
                if got != length:
                    raise OSError("short read")
            with concurrent.futures.ThreadPoolExecutor(threads) as pool:
                t0 = time.perf_counter()
                # Submission is chunked per worker slot so each buffer has one user.
                for start in range(0, reads, threads):
                    list(pool.map(one, range(start, min(reads, start + threads))))
                dt = time.perf_counter() - t0
            print(json.dumps({"arm": name, "seed": seed, "threads": threads, "reads": reads,
                              "bytes": reads * length, "seconds": round(dt, 4),
                              "gb_per_s": round(reads * length / dt / 1e9, 3)}), flush=True)
    finally:
        os.close(fd)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
