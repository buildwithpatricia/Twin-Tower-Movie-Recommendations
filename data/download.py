"""
Download and extract the MovieLens 1M dataset.

MovieLens 1M contains:
  - 1,000,209 ratings from 6,040 users on 3,706 movies
  - Ratings are on a 1-5 star scale
  - Each user has rated at least 20 movies

Source: https://grouplens.org/datasets/movielens/1m/
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192

    with open(dest_path, "wb") as f, tqdm(
        desc=f"Downloading {dest_path.name}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def download_movielens_1m(data_dir: str = "data/raw") -> Path:
    """
    Download and extract MovieLens 1M to data_dir.

    Args:
        data_dir: Directory to store the raw dataset.

    Returns:
        Path to the extracted ml-1m directory.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = data_dir / "ml-1m"
    if extract_dir.exists():
        print(f"Dataset already exists at {extract_dir}. Skipping download.")
        return extract_dir

    zip_path = data_dir / "ml-1m.zip"

    # Download
    print(f"Downloading MovieLens 1M from {ML1M_URL} ...")
    download_file(ML1M_URL, zip_path)

    # Extract
    print(f"Extracting to {data_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    zip_path.unlink()  # remove zip to save space
    print(f"Done. Data available at {extract_dir}")
    return extract_dir


if __name__ == "__main__":
    download_movielens_1m()
