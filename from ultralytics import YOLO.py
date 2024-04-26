import cv2
from ultralytics import YOLO
from sort import Sort

# Initialize the SORT tracker
tracker = Sort()

# Perform object tracking
tracked_objects = tracker.update(detections)
from sort import Sort

# Initialize YOLOv8 object detector
model = YOLO("C:/Users/ASUS/Downloads/hhhhh/best (1).pt")

# Initialize SORT tracker
tracker = Sort()

# Open video capture
cap = cv2.VideoCapture('C:/Users/ASUS/Downloads/hhhhh/test.mp4')

while True:
    # Read frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv8
    results = model(frame)

    # Extract detections and format them for tracking (e.g., [xmin, ymin, xmax, ymax, class_id, confidence])
    detections = [[box[0], box[1], box[2], box[3], box[5], box[4]] for box in results.xyxy[0]]

    # Perform object tracking with SORT
    tracked_objects = tracker.update(detections)

    # Visualize tracked objects on the frame
    for obj in tracked_objects:
        xmin, ymin, xmax, ymax, class_id, _ = obj
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, f'Class: {class_id}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
