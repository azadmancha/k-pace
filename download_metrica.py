"""Metrica Sports open tracking dataset downloader."""

import os
import sys
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"

FILES = [
    # Game 1
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    # Game 2
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    # Game 3
    "Sample_Game_3/Sample_Game_3_events.json",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
]


def download_file(rel_path: str, target_dir: str) -> None:
    filename = os.path.basename(rel_path)
    target_path = os.path.join(target_dir, filename)
    url = BASE_URL + rel_path

    if os.path.exists(target_path):
        return

    print(f"Downloading {filename}...")

    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100.0, block_num * block_size * 100.0 / total_size)
            sys.stdout.write(f"\r  {filename}: {percent:.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
        print()
    except Exception as e:
        print(f"\nFailed to download {filename}: {e}")


def main():
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(target_dir, exist_ok=True)

    for f in FILES:
        download_file(f, target_dir)


if __name__ == "__main__":
    main()
