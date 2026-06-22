from ultralytics import YOLO 

import cv2
import matplotlib.pyplot as plt 

model = YOLO('weights/best.pt') 

result=model.predict(source='image.png',save=True,conf=0.25)
for r in result:
    annotated_frame = r.plot()
img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10)) 
plt.imshow(img_rgb)
plt.axis('off')
plt.show()