import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8")
    parser.add_argument(
        "--webcam-resolution", 
        type=int, 
        default=[1280, 720], 
        nargs=2
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cap = cv2.VideoCapture(1)  # Change to 0 for primary webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.webcam_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.webcam_resolution[1])

    model = YOLO("yolov8l.pt")
    bbox_annotator = sv.BoxAnnotator(thickness=2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        person_detections = detections[detections.class_id == 0]
        frame = bbox_annotator.annotate(scene=frame, detections=person_detections)

        for i, (x, y, x2, y2) in enumerate(person_detections.xyxy):
            label = f"Person {person_detections.confidence[i]:.2f}"
            cv2.putText(
                frame, label, (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        cv2.imshow("YOLOv8 - People Detection", frame)

        if cv2.waitKey(30) == 27:  # Press 'Esc' to exit the webcam
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
