import shutil
import random
from pathlib import Path

random.seed(42)

IMAGES_DIR = Path("data/images")
LABELS_DIR = Path("data/labels")
OUTPUT_DIR = Path("data/dataset")

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

def main():
    image_files = sorted(IMAGES_DIR.glob("*.png"))
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": image_files[:n_train],
        "val":   image_files[n_train:n_train + n_val],
        "test":  image_files[n_train + n_val:]
    }

    for split_name, files in splits.items():
        img_out = OUTPUT_DIR / split_name / "images"
        lbl_out = OUTPUT_DIR / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            lbl_path = LABELS_DIR / (img_path.stem + ".txt")
            shutil.copy(img_path, img_out / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_out / lbl_path.name)

        print(f"{split_name}: {len(files)} images")

    print(f"\nDone. Dataset saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()