from ultralytics import YOLO

MODEL = "yolov8s.pt"
DATA  = "data/dataset.yaml"

def main():
    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="runs",
        name="yolov8s_no_person",
        exist_ok=True,
        verbose=True
    )

if __name__ == "__main__":
    main()