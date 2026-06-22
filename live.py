import cv2
from ultralytics import YOLO

model=YOLO('weights/best.pt')

cap=cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame=cap.read() 
    if not ret:break
    
    result=model.predict(source=frame,stream=True,conf=0.70)
    for r in result:
        annotated_frame=r.plot()
    cv2.imshow('Real time obstacle detection',annotated_frame)
    
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()