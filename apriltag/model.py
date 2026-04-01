from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

idx = 0
cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {idx}. Check permissions or try another index.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read frame (ret=False). Camera may be busy or permission denied.")
        break

    results = model.predict(frame, imgsz=320, verbose=False)
    cv2.imshow("YOLO", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()