# YOLOv8 Fruit Detection (Wrapstation Test - Task 1)

This repository contains my solution for Task 1 of the Wrapstation Fullstack Developer technical test. It implements a YOLOv8-based object detection model to identify and draw bounding boxes around fruits.

## Requirements
- Python >= 3.8
- `ultralytics`
- `opencv-python`

## Setup & Installation
Simply install the required dependencies via pip:
```bash
pip install ultralytics opencv-python
```
*(If you're on Windows and python is not recognized, use `py -m pip install ultralytics opencv-python`)*

## How to Use

### 1. Training the Model
To start the training process, run:
```bash
python train.py
```
The best weights will be saved automatically to `runs/detect/fruit_detection_model/weights/best.pt`.

**⚠️ Important Note Regarding the Dataset:** 
The Kaggle dataset link provided in the test instructions points to a "Multi-Class Classification" dataset. Since it lacks `.txt` bounding box annotations, running the script with the provided dataset will result in a `No labels found` error from YOLO. The script is fully functional and will train successfully once a proper YOLO Object Detection formatted dataset is provided.

### 2. Running Inference
Once the model is trained (`best.pt` is generated), you can run inference to see the model in action:
```bash
python inference.py
```
This will open an OpenCV window displaying the detected fruits with their bounding boxes and confidence scores.
- Press **any key** to skip to the next image.
- Press **q** to exit the preview.
