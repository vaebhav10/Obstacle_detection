from pathlib import Path

import cv2
from ultralytics import YOLO
import time 
BASE_DIR = Path(__file__).resolve().parent

model = YOLO(BASE_DIR / "weights/best.pt")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(6, 480)

start = time.time()

while True :
    success , img = cap.read()
    results = model(img, stream = True)
    for r in results:
        frame = r.plot()
        
    end = time.time()
    fps = 1/(end - start)
    start = end
    h, w = frame.shape[:2]
    
    cv2.putText(
        frame,
        f'fps: {int(fps)}',(5, h-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  
        (0, 255, 0),
        2
    )
    cv2.imshow("Detecting...", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
