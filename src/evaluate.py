from pathlib import Path
from ultralytics import YOLO

WEIGHTS  = Path("runs/yolov8s_no_person/weights/best.pt")
DATA     = Path("data/dataset.yaml")
IMG_SIZE = 640
BATCH    = 16
DEVICE   = 0  # 0 = GPU, "cpu" = CPU


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA}")

    model = YOLO(str(WEIGHTS))

    print(f"Evaluating on test split: {DATA}")
    print(f"Weights: {WEIGHTS}\n")

    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project="runs",
        name="eval_test",
        exist_ok=True,
        verbose=True,
    )

    # --- summary ----------------------------------------------------------
    mp   = metrics.box.mp        # mean Precision
    mr   = metrics.box.mr        # mean Recall
    map50    = metrics.box.map50     # mAP @ IoU=0.50
    map5095  = metrics.box.map       # mAP @ IoU=0.50:0.95

    print("\n=== Test-split results ===")
    print(f"  Precision  (mean): {mp:.4f}")
    print(f"  Recall     (mean): {mr:.4f}")
    print(f"  mAP50            : {map50:.4f}")
    print(f"  mAP50-95         : {map5095:.4f}")
    print("\nFull results saved to runs/eval_test/")


if __name__ == "__main__":
    main()