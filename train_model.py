from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8m.pt')
    
    result = model.train(
        data='ROD-Dataset/data.yaml',
        epochs=15,
        imgsz=640,
        workers=8,
        device='cuda'
    )
    