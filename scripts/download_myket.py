"""
Download Myket dataset from HuggingFace and export to data/myket.csv.

Usage:
    uv run python scripts/download_myket.py
    uv run python scripts/download_myket.py --output data/myket.csv
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download Myket dataset")
    parser.add_argument("--output", default="data/myket.csv", help="Output CSV path")
    parser.add_argument("--split", default="train", help="Dataset split to download")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading erfanloghmani/myket-android-application-recommendation ({args.split})...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset(
        "erfanloghmani/myket-android-application-recommendation",
        split=args.split,
    )

    print(f"Downloaded {len(ds):,} rows. Columns: {ds.column_names}")

    # Normalize column names → user_id, app_id
    df = ds.to_pandas()
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "user" in lower:
            col_map[col] = "user_id"
        elif "app" in lower or "item" in lower:
            col_map[col] = "app_id"
    if col_map:
        df = df.rename(columns=col_map)

    # Keep only the columns we need
    keep = [c for c in ["user_id", "app_id"] if c in df.columns]
    if len(keep) < 2:
        print(f"ERROR: Could not find user_id/app_id columns. Got: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    df = df[keep].drop_duplicates()
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df):,} interactions → {out_path}")
    print(f"  Unique users : {df['user_id'].nunique():,}")
    print(f"  Unique apps  : {df['app_id'].nunique():,}")


if __name__ == "__main__":
    main()
