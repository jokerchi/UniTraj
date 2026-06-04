#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["time", "matched_latitude", "matched_longitude"]
PARQUET_COLUMNS = ["time", "latitude", "longitude"]


def collect_files(input_path: str, input_format: str):
    suffix = "*.csv" if input_format == "csv" else "*.pkl"
    path = Path(input_path).expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob(suffix))
    return sorted(Path(p).expanduser() for p in glob.glob(input_path, recursive=True))


def read_trajectory_csv(path: Path):
    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError(f"No valid rows in {path} after filtering required columns.")

    times = pd.to_datetime(
        df["time"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    ).reset_index(drop=True)
    times.name = "time"
    trajectory = (
        df[["matched_latitude", "matched_longitude"]]
        .astype("float32")
        .values
        .tolist()
    )
    return {"time": times, "trajectory": trajectory}


def iter_pickle_records(path: Path):
    df = pd.read_pickle(path)
    for _, row in df.iterrows():
        if "trajectory" in row and "time" in row:
            yield {"time": row["time"], "trajectory": row["trajectory"]}
        elif all(column in row for column in PARQUET_COLUMNS):
            yield {
                "time": row["time"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
            }
        else:
            raise ValueError(
                f"Unsupported pickle schema in {path}. Expected columns "
                "'time' + 'trajectory' or 'time' + 'latitude' + 'longitude'."
            )


def iter_input_records(files, input_format):
    for path in files:
        if input_format == "csv":
            yield read_trajectory_csv(path)
        elif input_format == "pickle":
            yield from iter_pickle_records(path)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")


def to_unix_seconds(values):
    arr = np.asarray(values)
    if arr.size == 0:
        return []

    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[s]").astype("int64").tolist()

    if np.issubdtype(arr.dtype, np.number):
        numeric = arr.astype("int64", copy=False)
        max_abs = int(np.max(np.abs(numeric)))
        if max_abs > 10**17:
            numeric = numeric // 1_000_000_000
        elif max_abs > 10**14:
            numeric = numeric // 1_000_000
        elif max_abs > 10**11:
            numeric = numeric // 1_000
        return numeric.tolist()

    parsed = pd.to_datetime(values, errors="raise")
    return (parsed.astype("int64") // 1_000_000_000).astype("int64").tolist()


def normalize_for_parquet(record):
    if "latitude" in record and "longitude" in record:
        latitude = np.asarray(record["latitude"], dtype=np.float32)
        longitude = np.asarray(record["longitude"], dtype=np.float32)
    else:
        trajectory = np.asarray(record["trajectory"], dtype=np.float32)
        if trajectory.ndim != 2 or trajectory.shape[1] < 2:
            raise ValueError("trajectory must have shape [N, 2] with [lat, lon].")
        latitude = trajectory[:, 0]
        longitude = trajectory[:, 1]

    time = to_unix_seconds(record["time"])
    if len(time) != len(latitude) or len(latitude) != len(longitude):
        raise ValueError(
            "time, latitude, and longitude must have the same trajectory length."
        )

    return {
        "time": time,
        "latitude": latitude.astype(np.float32, copy=False).tolist(),
        "longitude": longitude.astype(np.float32, copy=False).tolist(),
    }


def normalize_for_pickle(record):
    if "trajectory" in record:
        return record

    latitude = np.asarray(record["latitude"], dtype=np.float32)
    longitude = np.asarray(record["longitude"], dtype=np.float32)
    trajectory = np.stack([latitude, longitude], axis=1).tolist()
    return {"time": record["time"], "trajectory": trajectory}


def write_pickle(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_pickle(out_path)


def write_parquet(records, out_path: Path, row_group_size=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("time", pa.list_(pa.int64())),
            pa.field("latitude", pa.list_(pa.float32())),
            pa.field("longitude", pa.list_(pa.float32())),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        row_group_size=row_group_size or min(len(records), 1024),
    )


def write_metadata(out_dir: Path, shards):
    metadata = {
        "format": "parquet",
        "total_trajectories": int(sum(item["num_trajectories"] for item in shards)),
        "num_shards": len(shards),
        "shards": shards,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def flush_shard(records, output_format, out_path, shard_idx, row_group_size):
    if output_format == "pickle":
        if shard_idx is None:
            write_pickle([normalize_for_pickle(r) for r in records], out_path)
            return out_path
        shard_path = out_path / f"trajectories_{shard_idx:06d}.pkl"
        write_pickle([normalize_for_pickle(r) for r in records], shard_path)
        return shard_path

    shard_path = out_path / f"trajectories_{shard_idx:06d}.parquet"
    write_parquet(
        [normalize_for_parquet(r) for r in records],
        shard_path,
        row_group_size=row_group_size,
    )
    return shard_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert trajectory CSV or pickle data to UniTraj pickle or "
            "streaming-friendly Parquet shards."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file, directory, or glob pattern.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output .pkl file path for unsharded pickle, or output directory "
            "when writing shards/parquet."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["csv", "pickle"],
        default="csv",
        help="Input data format.",
    )
    parser.add_argument(
        "--output-format",
        "--format",
        dest="output_format",
        choices=["pickle", "parquet"],
        default="pickle",
        help="Output data format.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help=(
            "Number of trajectories per shard. Defaults to 5000 for Parquet "
            "and keeps the original single-file behavior for pickle."
        ),
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=1024,
        help="Parquet row group size.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print progress every N trajectories (set 0 to disable).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    files = collect_files(args.input, args.input_format)
    if not files:
        raise SystemExit(f"No {args.input_format} files found for input: {args.input}")

    output_format = args.output_format
    shard_size = args.shard_size
    if output_format == "parquet" and shard_size is None:
        shard_size = 5000

    out_path = Path(args.output).expanduser()
    if output_format == "parquet" or shard_size:
        out_path.mkdir(parents=True, exist_ok=True)

    records = []
    shards = []
    shard_idx = 0
    total = 0

    for record in iter_input_records(files, args.input_format):
        records.append(record)
        total += 1
        if args.log_every and total % args.log_every == 0:
            print(f"Processed {total} trajectories")

        if shard_size and len(records) >= shard_size:
            shard_idx += 1
            shard_path = flush_shard(
                records, output_format, out_path, shard_idx, args.row_group_size
            )
            shards.append(
                {
                    "file": shard_path.name,
                    "num_trajectories": len(records),
                }
            )
            records = []

    if records:
        if shard_size:
            shard_idx += 1
            shard_path = flush_shard(
                records, output_format, out_path, shard_idx, args.row_group_size
            )
            shards.append(
                {
                    "file": shard_path.name,
                    "num_trajectories": len(records),
                }
            )
        else:
            flush_shard(records, output_format, out_path, None, args.row_group_size)

    if output_format == "parquet":
        write_metadata(out_path, shards)

    print(f"Done. Converted {total} trajectories.")


if __name__ == "__main__":
    main()


# CSV to pickle, original behavior:
# python convert_matched_csv_to_pickle.py --input /path/to/csv_dir --output data/custom.pkl
#
# Existing pickle shards to Parquet shards:
# python convert_matched_csv_to_pickle.py --input data/shards/train --input-format pickle --output data/parquet/train --output-format parquet --shard-size 5000

# python convert_matched_csv_to_pickle.py \
#   --input /path/to/csv_dir \
#   --output /path/to/parquet_dir \
#   --output-format parquet \
#   --shard-size 5000

# csv文件转换成为parquet文件
# python convert_matched_csv_to_pickle.py --input /home/stu252261/dataset/trajectory/data/yuanshao/OpenTrace/Trajectory --output ~/UniTraj/data/shards --output-format parquet --shard-size 50000