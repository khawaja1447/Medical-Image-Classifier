"""
Download the Kaggle Chest X-Ray Pneumonia dataset.

Prerequisites
-------------
1.  pip install kaggle
2.  Create your API key at https://www.kaggle.com/settings → API → Create New Token
3.  Place kaggle.json in ~/.kaggle/   (Linux/Mac) or %USERPROFILE%\\.kaggle\\  (Windows)
4.  chmod 600 ~/.kaggle/kaggle.json   (Linux/Mac only)

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --dest data/chest_xray
"""

import argparse
import zipfile
from pathlib import Path

DATASET = "paultimothymooney/chest-xray-pneumonia"
DEFAULT_DEST = Path("data")


def download(dest: Path = DEFAULT_DEST):
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "chest-xray-pneumonia.zip"

    if zip_path.exists():
        print(f"Archive already exists at {zip_path} — skipping download.")
    else:
        print(f"Downloading '{DATASET}' from Kaggle …")
        import kaggle
        api = kaggle.KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET, path=str(dest), unzip=False)
        print("Download complete.")

    print("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink(missing_ok=True)

    # Normalise folder name
    extracted = dest / "chest_xray"
    candidates = ["chest_xray", "chest-xray-pneumonia", "ChestXRay2017"]
    for name in candidates:
        candidate = dest / name
        if candidate.exists() and candidate != extracted:
            candidate.rename(extracted)
            break

    print(f"\nDataset ready at: {extracted.resolve()}")
    print("-" * 40)
    for split in ("train", "val", "test"):
        split_dir = extracted / split
        if not split_dir.exists():
            continue
        counts = {
            cls: len(list((split_dir / cls).glob("*")))
            for cls in ("NORMAL", "PNEUMONIA")
            if (split_dir / cls).exists()
        }
        total = sum(counts.values())
        print(f"  {split:5s}: {total:6,} images  {counts}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle Chest X-Ray dataset")
    parser.add_argument(
        "--dest", default=str(DEFAULT_DEST),
        help="Directory to save the dataset (default: data/)",
    )
    args = parser.parse_args()
    download(Path(args.dest))


if __name__ == "__main__":
    main()
