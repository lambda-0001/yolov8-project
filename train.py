from ultralytics import YOLO

# Load a model
model = YOLO("./weights/yolov8n.pt")

if __name__ == "__main__":
    train_results = model.train(
        data="./ultralytics/cfg/datasets/train2025.yaml",
        batch=1,
        imgsz=120,
        device="cpu",
        optimizer="SGD",
        epochs=100,
        hsv_h=0.3,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.5,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,
        erasing=0.5,
        auto_augment=None,
    )
