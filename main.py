import cv2
import argparse
from ultralytics import YOLO
import supervision as sv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8")
    parser.add_argument(
        "--webcam-resolution", 
        type=int, 
        default=[1280,720], 
        nargs = 2
    )
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    cap = cv2.VideoCapture(1) # change to 0 for primary webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.webcam_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.webcam_resolution[1])

    model = YOLO("yolov8l.pt")
    bbox = sv.BoxAnnotator(
        thickness=2,
    )

    while True:
        ret, frame = cap.read()
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = []

        frame = bbox.annotate(scene = frame, detections = detections)
        cv2.imshow("yolov8", frame)


        if cv2.waitKey(30) == 27:
            break

if __name__ == "__main__":
    main()