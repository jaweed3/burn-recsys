"""
Download MovieLens 1M and export to data/movielens.csv.

Usage:
    uv run python scripts/download_movielens.py
    uv run python scripts/download_movielens.py --output data/movielens.csv
"""
import argparse
import urllib.request
import zipfile
import io
from pathlib import Path


ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def main():
    parser = argparse.ArgumentParser(description="Download MovieLens 1M dataset")
    parser.add_argument("--output", default="data/movielens.csv", help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MovieLens 1M from {ML1M_URL}...")
    with urllib.request.urlopen(ML1M_URL) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open("ml-1m/ratings.dat") as f:
            lines = f.read().decode("latin-1").splitlines()

    rows = []
    for line in lines:
        parts = line.split("::")
        if len(parts) >= 2:
            rows.append((int(parts[0]), int(parts[1])))

    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "app_id"])
        writer.writerows(rows)

    unique_users = len({r[0] for r in rows})
    unique_items = len({r[1] for r in rows})
    print(f"Saved {len(rows):,} interactions → {out_path}")
    print(f"  Unique users : {unique_users:,}")
    print(f"  Unique items : {unique_items:,}")


if __name__ == "__main__":
    main()
