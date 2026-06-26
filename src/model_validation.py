from ultralytics import YOLO 
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

model = YOLO(BASE_DIR/"weights/best.pt")

test_metrics = model.val(
    data=BASE_DIR/'ROD-Dataset/data.yaml',
    split='test',
    device='cuda',
    imgsz=640
    )
print(f"mAP50    :{test_metrics.box.map50}")
print(f"mAP50-95 : {test_metrics.box.map}")