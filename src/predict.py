"""
predict.py  --  PPE helmet detection with violation alerts.

Usage:
    # single image
    python src/predict.py --source path/to/image.jpg

    # folder of images
    python src/predict.py --source path/to/images/

    # lower confidence threshold (better Recall for 'head')
    python src/predict.py --source path/to/images/ --conf 0.20

Outputs:
    runs/predict/<source_name>/
        *.jpg          -- annotated images
        violations.txt -- list of images with violations
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------------------------
WEIGHTS     = Path("runs/yolov8s_no_person/weights/best.pt")
OUTPUT_DIR  = Path("runs/predict")
IMG_SIZE    = 640
DEVICE      = 0       # 0 = GPU, "cpu" = CPU

CLASS_NAMES = {0: "helmet", 1: "head"}

# BGR colors for bounding boxes
COLOR_HELMET    = (0, 200, 0)    # green
COLOR_HEAD      = (0, 0, 220)    # red
COLOR_VIOLATION = (0, 0, 255)    # bright red  (head without helmet)

# A head bbox is considered "covered" if its center falls inside
# a helmet bbox expanded by this fraction of the helmet size.
IOU_COVER_MARGIN = 0.15
# -------------------------------------------------------------------


def box_center(box):
    """Return (cx, cy) of a xyxy box."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def is_covered(head_box, helmet_boxes, margin=IOU_COVER_MARGIN):
    """
    Return True if the center of head_box lies within any helmet_box
    expanded by `margin` on each side.
    """
    cx, cy = box_center(head_box)
    for hb in helmet_boxes:
        x1, y1, x2, y2 = hb
        w, h = x2 - x1, y2 - y1
        if (x1 - margin * w <= cx <= x2 + margin * w and
                y1 - margin * h <= cy <= y2 + margin * h):
            return True
    return False


def draw_box(img, box, label, color, conf):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, text, (x1 + 1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def process_image(img_path, model, conf_thresh, out_dir):
    """
    Run inference on one image.
    Returns True if at least one violation is detected.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Cannot read {img_path}, skipping.")
        return False

    results = model.predict(
        source=str(img_path),
        imgsz=IMG_SIZE,
        conf=conf_thresh,
        device=DEVICE,
        verbose=False,
    )[0]

    boxes  = results.boxes.xyxy.cpu().numpy()    # (N, 4)
    confs  = results.boxes.conf.cpu().numpy()    # (N,)
    labels = results.boxes.cls.cpu().numpy().astype(int)  # (N,)

    helmet_boxes = boxes[labels == 0]
    head_indices = np.where(labels == 1)[0]

    violation = False

    for i, (box, conf, cls) in enumerate(zip(boxes, confs, labels)):
        name = CLASS_NAMES[cls]
        if cls == 1:  # head
            covered = is_covered(box, helmet_boxes)
            if not covered:
                violation = True
                draw_box(img, box, "NO HELMET", COLOR_VIOLATION, conf)
            else:
                draw_box(img, box, name, COLOR_HEAD, conf)
        else:  # helmet
            draw_box(img, box, name, COLOR_HELMET, conf)

    # stamp violation banner
    if violation:
        cv2.rectangle(img, (0, 0), (220, 30), (0, 0, 200), -1)
        cv2.putText(img, "! VIOLATION DETECTED", (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    return violation


def collect_images(source):
    source = Path(source)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in exts)


def main():
    parser = argparse.ArgumentParser(description="PPE inference + violation detection")
    parser.add_argument("--source", required=True,
                        help="Image file or folder")
    parser.add_argument("--conf",   type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--weights", default=str(WEIGHTS),
                        help="Path to model weights")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    images = collect_images(args.source)
    if not images:
        raise ValueError(f"No images found at: {args.source}")

    # output dir named after source
    source_name = Path(args.source).stem
    out_dir = OUTPUT_DIR / source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    print(f"Source   : {args.source}  ({len(images)} images)")
    print(f"Conf     : {args.conf}")
    print(f"Output   : {out_dir}\n")

    violation_files = []
    for img_path in images:
        has_violation = process_image(img_path, model, args.conf, out_dir)
        status = "VIOLATION" if has_violation else "ok"
        print(f"  {img_path.name:40s}  [{status}]")
        if has_violation:
            violation_files.append(img_path.name)

    # save violation list
    vio_txt = out_dir / "violations.txt"
    vio_txt.write_text("\n".join(violation_files))

    print(f"\nViolations : {len(violation_files)} / {len(images)}")
    print(f"Results    : {out_dir}")
    print(f"Violations : {vio_txt}")


if __name__ == "__main__":
    main()