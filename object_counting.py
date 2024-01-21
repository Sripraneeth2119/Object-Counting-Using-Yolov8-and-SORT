import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *

# VideoCapture and model development
video_path = r'E:\Personal Project\Data\veh2.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('yolov8n.pt')
x1,y1, x2,y2, x3,y3, x4,y4 = 1160,950, 1860,950, 1860,960, 1160,960
x5,y5, x6,y6, x7,y7, x8,y8 = 340,768, 940,768, 940,778, 340,778

#Classes names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Defining the roi
rightLane = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
rightLane_line = np.array([rightLane[0],rightLane[1]]).reshape(-1)
leftLane = np.array([[x5,y5],[x6,y6],[x7,y7],[x8,y8]], np.int32)
leftLane_line = np.array([leftLane[0],leftLane[1]]).reshape(-1)

# Initiating the Tracker
tracker = Sort()
rightLane_counter = []
leftLane_counter = []

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (1920,1080))
    results = model(frame)
    current_detections = np.empty([0,5])

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            
            #Bounding Box
            cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if class_detect == 'car' or class_detect == 'truck' or class_detect == 'bus'\
                    and conf > 60:
                detections = np.array([x1,y1,x2,y2,conf])
                current_detections = np.vstack([current_detections,detections])

    # Creating ROI line
    cv2.polylines(frame,[rightLane], isClosed=True, color=(0, 0, 255), thickness=8)
    cv2.polylines(frame,[leftLane], isClosed=True, color=(0, 0, 255), thickness=8)

    track_results = tracker.update(current_detections)
    for result in track_results:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2 -40


        if rightLane_line[0] < cx < rightLane_line[2] and rightLane_line[1] - 20 < cy < rightLane_line[1] + 20:
            if rightLane_counter.count(id) == 0:
                rightLane_counter.append(id)
        if leftLane_line[0] < cx < leftLane_line[2] and leftLane_line[1] - 20 < cy < leftLane_line[1] + 20:
            if leftLane_counter.count(id) == 0:
                leftLane_counter.append(id) 

        cvzone.putTextRect(frame, f'Right LANE Vehicles ={len(rightLane_counter)}', [1300, 99], thickness=4,scale=2.3, border=2)
        cvzone.putTextRect(frame, f'Left LANE Vehicles ={len(leftLane_counter)}', [1300, 140], thickness=4,scale=2.3, border=2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows