from ultralytics import YOLO 
import cv2
import matplotlib.pyplot as plt 

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

model = YOLO(BASE_DIR/"weights/best.pt")

path =input('Enter image path for detection :')

result=model.predict(source=BASE_DIR/path,save=True,conf=0.30)
for r in result:
    annotated_frame = r.plot()
img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10)) 
plt.imshow(img_rgb)
plt.axis('off')
plt.show()