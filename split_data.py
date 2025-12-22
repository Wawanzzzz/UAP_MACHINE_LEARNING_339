import shutil
import random
from pathlib import Path

# =========================================================
# KONFIGURASI PATH (AMAN BERDASARKAN LOKASI FILE)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent        # folder src/
SOURCE_DIR = BASE_DIR / "data"                    # src/data/
OUTPUT_DIR = BASE_DIR / "src/data_split"              # src/data_split/

# =========================================================
# KONFIGURASI SPLIT DATA
# =========================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# =========================================================
# FUNGSI MEMBUAT STRUKTUR FOLDER OUTPUT
# =========================================================
def make_dirs():
    for split in ["train", "val", "test"]:
        for cls in ["biasa", "skena"]:
            dir_path = OUTPUT_DIR / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)

# =========================================================
# FUNGSI SPLIT DAN COPY DATA
# =========================================================
def split_and_copy(class_name):
    src_dir = SOURCE_DIR / class_name
    images = [
        f for f in src_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        for file in files:
            dst_dir = OUTPUT_DIR / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst = dst_dir / file.name
            shutil.copy(file, dst)

    print(
        f"{class_name}: total={total}, "
        f"train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )

# =========================================================
# MAIN
# =========================================================
def main():
    make_dirs()
    for cls in ["biasa", "skena"]:
        split_and_copy(cls)

    print("\n✅ Dataset berhasil dibagi dan disimpan di 'src/data_split/'")

if __name__ == "__main__":
    main()
