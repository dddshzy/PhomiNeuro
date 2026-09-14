#!/usr/bin/env python3
"""Create approximately load-balanced head lists for parallel MCX workers."""

from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import repro_config as RC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=RC.STANDARD_DATASET_DIR)
    parser.add_argument("--output-dir", default=os.path.join(RC.WORK_DIR, "shards"))
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--num-sharm", type=int, default=196)
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be positive")

    scatterbrains = sorted(
        os.path.basename(path).split("_")[0]
        for path in glob.glob(os.path.join(args.dataset_dir, "scb*.mat"))
    )
    brainweb = sorted(
        os.path.basename(path).split("_")[0]
        for path in glob.glob(os.path.join(args.dataset_dir, "bw*.mat"))
    )
    sharm = [f"sh{i:03d}" for i in range(1, args.num_sharm + 1)]
    heads = scatterbrains + brainweb + sharm

    costs = {
        head: 64 if head.startswith(("scb", "bw")) else 34 for head in heads
    }
    shards = [[] for _ in range(args.num_shards)]
    loads = [0] * args.num_shards
    for head in sorted(heads, key=lambda item: -costs[item]):
        index = min(range(args.num_shards), key=loads.__getitem__)
        shards[index].append(head)
        loads[index] += costs[head]

    os.makedirs(args.output_dir, exist_ok=True)
    for index, shard in enumerate(shards):
        output = os.path.join(args.output_dir, f"sim_shard{index}.txt")
        with open(output, "w", encoding="utf-8") as stream:
            stream.write(" ".join(shard) + "\n")
        print(f"shard{index}: {len(shard)} heads, approximately {loads[index]} scenes")
    print(f"total: {len(heads)} heads, approximately {sum(loads)} scenes")


if __name__ == "__main__":
    main()
