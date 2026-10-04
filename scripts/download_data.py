"""Download the GHAW-H dataset tables into data/raw/.

The download is pinned to a fixed Hugging Face revision of GHAW-H v0.1.2
(identical to Zenodo record 10.5281/zenodo.22084012), and every file is
checked against its SHA-256 before it is kept.

Usage:
    uv run python scripts/download_data.py              # tables needed for RQ1
    uv run python scripts/download_data.py --with-lock  # also lock-file snapshots (~70 MB)
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "pavtch/GHAW-H"
REVISION = "9ccc840bd7e96fd90db3a16548d5482c2833c47d"  # GHAW-H v0.1.2
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data"

# table name -> SHA-256 of the parquet file at REVISION
TABLES = {
    "repository": "75bc7e5a838281f49403e682709775936568f781031ef94b01bec343148c6ae8",
    "source_markdown_file_history": "5c90e748b8e5ad59de088111ea91af6b06ec08789d70e0a7600e9273ca5a02a4",
    "source_markdown_file_snapshot": "b8049b2f05bdbce449f24dfef3ca70e1b135b339a17347ad9d88a98f4e8d94b9",
    "source_markdown_file_version": "88576175db0645f52f399279d6e263f35fc2c152fbc18f734f2eb28c05cc1b2f",
}
LOCK_TABLE = {
    "lock_file_snapshot": "0272d5b7b7e93d26d52116214dd167b5f3940d135f75474092e06bb434df77b3",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, expected: str) -> None:
    dest = RAW_DIR / f"{name}.parquet"
    if dest.exists() and sha256(dest) == expected:
        print(f"ok (cached)  {dest.name}")
        return
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(f"{BASE_URL}/{name}.parquet", tmp)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {name}: expected {expected}, got {actual}")
    tmp.replace(dest)
    print(f"ok           {dest.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-lock", action="store_true", help="also download lock_file_snapshot")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tables = TABLES | (LOCK_TABLE if args.with_lock else {})
    for name, expected in tables.items():
        fetch(name, expected)


if __name__ == "__main__":
    main()
