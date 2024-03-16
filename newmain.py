import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from imutils import face_utils
from imutils.video import VideoStream
import matplotlib.animation as animate
from matplotlib import style
import imutils
import dlib
import time
import argparse
import cv2
from playsound import playsound
import os
import csv
import numpy as np
from datetime import datetime
from scipy.spatial import distance as dist

style.use('fivethirtyeight')

# Function to create a directory if it doesn't exist
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# Function to calculate the slope between two points
def calculate_slope(point1, point2):
    if point2[0] - point1[0] != 0:  # Ensure denominator is not zero
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        return slope
    else:
        return None
    
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    MAR = (A + B + C) / 3.0
    return MAR

# Argument Parsing
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether Raspberry Pi camera shall be used or not")
args = vars(ap.parse_args())

# Constants and Counters
EAR_THRESHOLD = 0.3
CONSECUTIVE_FRAMES = 20
MAR_THRESHOLD = 14

BLINK_COUNT = 0
FRAME_COUNT = 0

# Loading Face Detector and Landmark Predictor
print("[INFO] Loading the predictor.....")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Facial Landmarks Indices
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Video Stream Initialization
print("[INFO] Loading Camera.....")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2)

# Dataset Directory Creation
assure_path_exists("dataset/")
count_sleep = 0
count_yawn = 0

# Lists to store data for accuracy graph
accuracy_data = []

# Data Logging Initialization
log_file_path = 'drowsiness_log.csv'
fieldnames = ['Timestamp', 'Blink Count', 'Drowsiness']

# Create a new CSV file or append to an existing one
if not os.path.isfile(log_file_path):
    with open(log_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
FACE_ANGLE_THRESHOLD_MAX = 122
FACE_ANGLE_THRESHOLD_MIN = 112

# Main Loop for Video Processing
while True:
    frame = vs.read()
    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # Calculate vector connecting midpoint of eyes and midpoint of mouth
        midpoint_eyes = np.mean([shape[42], shape[45]], axis=0)
        midpoint_mouth = np.mean([shape[62], shape[66]], axis=0)
        face_vector = midpoint_mouth - midpoint_eyes
        reference_vector = np.array([1, 0])
        # Calculate angle between face vector and reference vector
        face_angle = angle_between_vectors(reference_vector, face_vector)
        # print("Face angle:", face_angle)
        # Check if face angle exceeds threshold
        if face_angle > FACE_ANGLE_THRESHOLD_MAX or face_angle < FACE_ANGLE_THRESHOLD_MIN:
            cv2.putText(frame, "FACE NOT FACING FRONT!", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # playsound('a2.mp3')
            # playsound(None, True)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend]
        mouth = shape[mstart:mend]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        EAR = (leftEAR + rightEAR) / 2.0

        ts = dt.datetime.now().strftime('%H:%M:%S.%f')
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        MAR = mouth_aspect_ratio(mouth)
        if EAR < EAR_THRESHOLD:
            FRAME_COUNT += 1

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                count_sleep += 1
                cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # playsound('a1.mp3')
                # playsound(None, True)
                accuracy_data.append(1)  # 1 indicates drowsy prediction
                BLINK_COUNT += 1
                with open(log_file_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'Timestamp': ts, 'Blink Count': BLINK_COUNT, 'Drowsiness': 1})

        else:
            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                print("Drowsiness warning cleared.")
            FRAME_COUNT = 0
            accuracy_data.append(0)  # 0 indicates not drowsy prediction
            with open(log_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Timestamp': ts, 'Blink Count': BLINK_COUNT, 'Drowsiness': 0})

        if MAR > MAR_THRESHOLD:
            count_yawn += 1
            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
            cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
            accuracy_data.append(1)  # 1 indicates drowsy prediction
            with open(log_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Timestamp': ts, 'Blink Count': BLINK_COUNT, 'Drowsiness': 1})
        else:
            accuracy_data.append(0)  # 0 indicates not drowsy prediction

    # Display the frame
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# Plotting the Accuracy Graph
plt.plot(accuracy_data)
plt.title('Drowsiness Detection Accuracy Over Time')
plt.xlabel('Frame Number')
plt.ylabel('Prediction (1: Drowsy, 0: Not Drowsy)')
plt.show()

# Cleanup
cv2.destroyAllWindows()
vs.stop()
