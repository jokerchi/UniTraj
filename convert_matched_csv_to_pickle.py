#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["time", "matched_latitude", "matched_longitude"]


def collect_files(input_path: str):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    return sorted(Path(p) for p in glob.glob(input_path, recursive=True))


def read_trajectory_csv(path: Path):
    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError(f"No valid rows in {path} after filtering required columns.")
    times = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="raise").reset_index(drop=True)
    times.name = "time"
    trajectory = (
        df[["matched_latitude", "matched_longitude"]].astype(float).values.tolist()
    )
    return {"time": times, "trajectory": trajectory}


def write_pickle(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_pickle(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert per-trajectory CSV files to UniTraj pickle format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file, directory of CSVs, or glob pattern (e.g. data/**/*.csv).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .pkl file path, or output directory when using --shard-size.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of trajectories per shard; when set, --output is treated as a directory.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print progress every N files (set 0 to disable).",
    )
    args = parser.parse_args()

    files = collect_files(args.input)
    if not files:
        raise SystemExit(f"No CSV files found for input: {args.input}")

    if args.shard_size:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        shard_idx = 0
        for idx, path in enumerate(files, 1):
            records.append(read_trajectory_csv(path))
            if args.log_every and idx % args.log_every == 0:
                print(f"Processed {idx}/{len(files)} files")
            if len(records) >= args.shard_size:
                shard_idx += 1
                shard_path = out_dir / f"trajectories_{shard_idx:06d}.pkl"
                write_pickle(records, shard_path)
                records = []
        if records:
            shard_idx += 1
            shard_path = out_dir / f"trajectories_{shard_idx:06d}.pkl"
            write_pickle(records, shard_path)
    else:
        records = []
        for idx, path in enumerate(files, 1):
            records.append(read_trajectory_csv(path))
            if args.log_every and idx % args.log_every == 0:
                print(f"Processed {idx}/{len(files)} files")
        write_pickle(records, Path(args.output))


if __name__ == "__main__":
    main()



# python3 convert_matched_csv_to_pickle.py --input /home/stu252261/dataset/trajectory/data/yuanshao/OpenTrace/Trajectory --output ~/UniTraj/data/shards --shard-size 50000
# python3 convert_matched_csv_to_pickle.py --input /home/stu252261/dataset/trajectory/data/yuanshao/OpenTrace/Trajectory/9999989.csv --output ~/UniTraj/data/shards/test.pkl