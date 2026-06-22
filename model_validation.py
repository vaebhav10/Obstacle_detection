from ultralytics import YOLO 

model = YOLO('weights/best.pt') 

test_metrics = model.val(
    data='ROD-Dataset/data.yaml',
    split='test',
    device='cuda',
    imgsz=640
    )
print(f"mAP50    :{test_metrics.box.map50}")
print(f"mAP50-95 : {test_metrics.box.map}")