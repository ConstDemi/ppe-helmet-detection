"""
export.py  --  Export YOLOv8 best.pt to ONNX format.

Usage:
    python src/export.py

    # with dynamic batch size (useful for server deployment)
    python src/export.py --dynamic

    # with half-precision (smaller file, faster on GPU)
    python src/export.py --half

Output:
    runs/yolov8s_no_person/weights/best.onnx
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

WEIGHTS  = Path("runs/yolov8s_no_person/weights/best.pt")
IMG_SIZE = 640


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--weights", default=str(WEIGHTS),
                        help="Path to .pt weights (default: best.pt)")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE,
                        help="Input image size (default: 640)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Enable dynamic batch size axis")
    parser.add_argument("--half", action="store_true",
                        help="Export in FP16 (requires GPU)")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Simplify ONNX graph via onnx-simplifier (default: True)")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))

    print(f"Exporting  : {weights}")
    print(f"Image size : {args.imgsz}")
    print(f"Dynamic    : {args.dynamic}")
    print(f"Half (FP16): {args.half}")
    print(f"Simplify   : {args.simplify}\n")

    out_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        half=args.half,
        simplify=args.simplify,
    )

    out_path = Path(out_path)
    size_mb  = out_path.stat().st_size / (1024 ** 2)

    print(f"\nDone.")
    print(f"  ONNX model : {out_path}")
    print(f"  File size  : {size_mb:.1f} MB")
    print()
    print("To run inference with ONNX:")
    print("  pip install onnxruntime-gpu   # GPU")
    print("  pip install onnxruntime       # CPU only")
    print()
    print("Quick check (Ultralytics wrapper):")
    print(f"  yolo predict model={out_path} source=data/dataset/test/images")


if __name__ == "__main__":
    main()