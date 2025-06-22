# 👁️ Eye Drowsiness Detection System

An advanced **real-time drowsiness detection system** using YOLO object detection and facial landmark analysis. Features dual detection methods, audio alerts, and Streamlit deployment for driver safety applications.

---

## Features

- **Dual Detection Methods**: YOLO-based eye state detection + dlib facial landmarks
- ** Real-time Processing**: Live webcam monitoring with sub-second inference
- **Audio Alert System**: Pygame-powered alarm when drowsiness detected
- **Web Interface**: Streamlit app for easy deployment
- **High Accuracy**: 97% precision, 98.2% mAP50 on validation set
- **Customizable Alerts**: Audio warning system with threshold controls

---

## 📂 Repository Structure

Eye_Drowsiness_Detection/
├── data/
│   ├── train/
│   │   ├── open_eyes/
│   │   └── closed_eyes/
│   ├── valid/
│   │   ├── open_eyes/
│   │   └── closed_eyes/
│   ├── test/
│   │   ├── open_eyes/
│   │   └── closed_eyes/
│   ├── data.yaml  # Data configuration for training
├── model/
│   ├── last.pt   # Trained model file
│   └── shape_predictor.dat  # Pre-trained dlib shape predictor
├── src/
│   ├── detection.py 
│   ├── prediction.py  # Real-time detection script for webcam
├── app.py  # Streamlit or Flask web app for deployment
├── requirements.txt  # List of dependencies for the project
└── README.md  # Project documentation


---

## ⚙️ Technical Implementation

### YOLO Detection Method

from ultralytics import YOLO
import cv2

Load trained model
model = YOLO("models/best.pt")

Real-time detection
results = model.predict(frame, imgsz=640, conf=0.5)

Process detections
for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
if model.names[int(cls)] == "closed_eye":
# Trigger drowsiness alert
closed_eye_counter += 1


### 👁️ Facial Landmark Method

import dlib
from scipy.spatial import distance

def eye_aspect_ratio(eye):
A = distance.euclidean(eye, eye)
B = distance.euclidean(eye, eye)
C = distance.euclidean(eye, eye)
ear = (A + B) / (2.0 * C)
return ear

EAR threshold: 0.25
Frame threshold: 20 consecutive frames


### 🔊 Alert System

from pygame import mixer

mixer.init()
mixer.music.load("music/music.wav")

Trigger alert when drowsiness detected
if closed_eye_counter >= threshold:
mixer.music.play()
cv2.putText(frame, "ALERT! Drowsiness Detected", ...)


---

## 📊 Model Performance

### Training Results (100 Epochs)
| Metric | Score |
|--------|-------|
| **Precision** | 97.0% |
| **Recall** | 97.5% |
| **mAP50** | 98.2% |
| **mAP50-95** | 76.0% |

### Class-wise Performance
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **Closed Eye** | 100% | 95.0% | 99.5% | 75.7% |
| **Open Eye** | 94.0% | 100% | 96.9% | 76.2% |

### Inference Speed
- **Preprocessing**: 9.2ms
- **Inference**: 21.0ms  
- **Postprocessing**: 5.2ms
- **Total**: ~35ms per frame

---

## Getting Started

### 📋 Prerequisites
- Python 3.8+
- OpenCV
- YOLO (Ultralytics)
- dlib
- Streamlit

### Installation

pip install -r requirements.txt


### ⚙️ Setup Models
1. **YOLO Model**: Download trained `best.pt` to `models/`
2. **Shape Predictor**: Download `shape_predictor_68_face_landmarks.dat`

### Usage

#### YOLO Detection

python detection/detection.py


#### Facial Landmark Detection 

python detection/prediction.py


#### Streamlit Web App

streamlit run app.py

### Configuration Parameters

YOLO Detection
closed_eye_threshold = 15 # Consecutive frames
confidence_threshold = 0.5 # Detection confidence

Facial Landmarks
ear_threshold = 0.25 # Eye Aspect Ratio
frame_check = 20 # Consecutive frames


### 📱 Streamlit Interface

import streamlit as st

st.title("Driver Drowsiness Detection")
run = st.checkbox("Start Camera")

if run:
# Real-time detection logic
while True:
ret, frame = cap.read()
# Process frame and detect drowsiness
FRAME_WINDOW.image(processed_frame)


---

## Model Training

### 📊 Dataset Structure

data/
├── train/
│ ├── open_eyes/ # Open eye images
│ └── closed_eyes/ # Closed eye images
├── valid/
│ ├── open_eyes/
│ └── closed_eyes/
└── test/
├── open_eyes/
└── closed_eyes/


### Training Process

from ultralytics import YOLO

Initialize model
model = YOLO('yolov8n.pt')

Train model
results = model.train(
data='data.yaml',
epochs=100,
imgsz=640,
batch=16
)


---

## Detection Workflow

1. **Video Capture**: Initialize webcam stream
2. **Face Detection**: Locate faces in frame
3. **Eye Detection**: Identify open/closed eye states
4. **Threshold Check**: Count consecutive closed-eye frames
5. **Alert Trigger**: Sound alarm when threshold exceeded
6. **Reset Counter**: Reset when eyes reopen

---

## 📈 Performance Optimization

- **GPU Acceleration**: CUDA support for faster inference
- **Confidence Tuning**: Optimized detection thresholds
- **Frame Processing**: Efficient image preprocessing pipeline
- **Model Optimization**: Pruned model for real-time performance

---

