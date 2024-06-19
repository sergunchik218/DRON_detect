import cv2
import os
import time
from ultralytics import YOLO


model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(0)
output_dir = "imagedetect"
os.makedirs(output_dir, exist_ok=True)

# Основной цикл
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, classes=[0, 2, 4, 5, 14])
    detected = False
    for result in results:
        for detection in result.boxes.data:
            x1, y1, x2, y2, conf, cls = detection[:6]
            if conf > 0.5:
                detected = True
                label = model.names[int(cls)]
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    if detected:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = os.path.join(output_dir, f"detected_{timestamp}.png")
        cv2.imwrite(image_path, frame)
        print(f"Скриншот сохранен: {image_path}")
    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
