#  Automatic Number Plate Recognition (ANPR)

**Deployed App:** [https://numberplaterecog.streamlit.app/)

Real-Time Vehicle License Plate Detection and Alphanumeric Text Extraction System.

## 📖 Overview

This project is a high-efficiency computer vision pipeline designed to automate license plate localization and character recognition. By combining deep learning-based object detection with robust Optical Character Recognition (OCR), the system processes image and video streams to detect vehicles, crop their license plates, and extract the text with high accuracy.

## 🚀 Key Features

* **Real-Time Detection:** Utilizes an optimized **YOLOv8** model to precisely localize and bound license plates within busy frames.
* **Accurate Character Extraction:** Integrates **EasyOCR** with specialized image preprocessing to handle varying fonts, lighting conditions, and angles.
* **Modular Pipeline:** Structured with isolated modules for detection (`plate_rec.py`), utility functions (`utils.py`), and user interface orchestration.
* **Streamlit UI:** Features a clean, interactive dashboard (`app.py`) for file uploads, real-time bounding box visualization, and instant text output.

## 🛠️ Technical Stack

* **Core Language:** Python
* **Computer Vision & DL:** YOLOv8 (Ultralytics), OpenCV
* **Text Extraction:** EasyOCR
* **Web Interface:** Streamlit
* **Environment Management:** Virtualenv (`venv`)

## 📂 Repository Structure

```text
NUMBER PLATE DETECTION/
├── app.py            # Streamlit web application & interface
├── plate_rec.py      # Main pipeline for plate detection & recognition
├── utils.py          # Image processing and geometric helper functions
├── yolov8n.pt        # Pre-trained YOLOv8 nano weights for license plates
├── requirements.txt  # Python package dependencies
├── LICENSE           # Project licensing terms
└── .gitattributes    # Git configuration for large tracking files/LFS
