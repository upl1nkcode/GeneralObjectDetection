from ultralytics import YOLO
import cv2
from pathlib import Path

from generalTesting.TestScriptYolo import annotated


class ObjectDetector:
    def __init__(self, model_path = "yolov8n.pt"):
        """
        -> model_path is the dataset used for the ai detection algorithm, it is to be changed since it is not accurate
        :param model_path:
        """
        self.model = YOLO(model_path)
        print(f"✅ Model '{model_path}' loaded successfully.")

    def detect_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Cannot read image")
            return
        results = self.model(image)
        annotated = results[0].plot()
        cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Image Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_video(self, video_path):
        """Run detection on a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video at {video_path}")
            return

        cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video Detection", 960, 540)
        print("🎞 Press 'q' to quit video detection.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video ended or cannot read frame.")
                break

            results = self.model(frame)
            annotated = results[0].plot()

            # Resize dynamically to window
            window_name = "Video Detection"
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w > 0 and win_h > 0:
                annotated = cv2.resize(annotated, (win_w, win_h))

            cv2.imshow(window_name, annotated)

            # Add delay to match frame rate (~25 ms = ~40 FPS)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_webcam(self, cam_index=0):
        """Run detection from webcam"""
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("❌ Error: Cannot access webcam.")
            return

        cv2.namedWindow("Webcam Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam Detection", 960, 540)
        print("🎥 Press 'q' to quit webcam mode.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            annotated = results[0].plot()

            cv2.imshow("Webcam Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

