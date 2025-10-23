from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model (Nano = fastest)
model = YOLO("yolov8s.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Draw results on frame
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Object Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
