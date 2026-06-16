"""
Download 1m LiDAR-derived DEM tiles from USGS National Map.

Usage:
    python scripts/download_dem.py                         # uses default data/raw/dem/
    python scripts/download_dem.py --out data/raw/dem      # explicit output dir
    python scripts/download_dem.py --workers 4             # parallel downloads
    python scripts/download_dem.py --dry-run               # list files without downloading

URLs are read from scripts/dem_urls.txt (one URL per line).
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

DEFAULT_URL_FILE = Path(__file__).parent / "dem_urls.txt"
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "dem"


def download_tile(url: str, out_dir: Path, session: requests.Session) -> tuple[str, bool, str]:
    filename = Path(urlparse(url).path).name
    dest = out_dir / filename

    if dest.exists():
        return filename, True, "skipped (already exists)"

    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=filename,
                leave=False,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        return filename, True, "downloaded"
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return filename, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Download USGS 1m DEM tiles")
    parser.add_argument("--urls", type=Path, default=DEFAULT_URL_FILE, help="Text file with one URL per line")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--workers", type=int, default=2, help="Parallel download threads (keep low to be polite)")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    args = parser.parse_args()

    if not args.urls.exists():
        print(f"URL file not found: {args.urls}")
        print("Copy your data.txt from the shared project folder to scripts/dem_urls.txt")
        sys.exit(1)

    urls = [line.strip() for line in args.urls.read_text().splitlines() if line.strip() and not line.startswith("#")]
    print(f"Found {len(urls)} tiles")

    if args.dry_run:
        for url in urls:
            print(url)
        return

    args.out.mkdir(parents=True, exist_ok=True)

    failed = []
    with requests.Session() as session, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_tile, url, args.out, session): url for url in urls}
        with tqdm(total=len(urls), desc="Total tiles", unit="tile") as pbar:
            for future in as_completed(futures):
                filename, ok, msg = future.result()
                pbar.set_postfix_str(f"{filename}: {msg}")
                pbar.update(1)
                if not ok:
                    failed.append((filename, msg))

    print(f"\nDone. {len(urls) - len(failed)}/{len(urls)} tiles downloaded to {args.out}")
    if failed:
        print("\nFailed:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
