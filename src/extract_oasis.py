import tarfile
import pathlib
import sys
import time

def extract_oasis_archives(archives_dir: str = "data/OASIS_raw", output_dir: str = "data/OASIS-1"):
    src_path = pathlib.Path(archives_dir)
    dest_path = pathlib.Path(output_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    for i in range(1, 13):
        archive = src_path / f"oasis_cross-sectional_disc{i}.tar.gz"
        dest = dest_path / f"disc{i}"

        if dest.exists() and any(dest.iterdir()):
            print(f"[disc{i}] Already extracted, skipping.")
            continue

        if not archive.exists():
            print(f"[disc{i}] Archive not found: {archive}")
            continue

        t0 = time.time()
        print(f"[disc{i}] Extracting {archive.name}...")
        try:
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(dest_path)
            elapsed = time.time() - t0
            print(f"[disc{i}] Extracted in {elapsed:.1f}s")
        except Exception as e:
            print(f"[disc{i}] Extraction error: {e}", file=sys.stderr)

if __name__ == "__main__":
    extract_oasis_archives()
