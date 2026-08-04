import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
reader = easyocr.Reader(['en'], gpu=False)

def draw_bounding_box(img, x, y, w, h, text=""):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    if text:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x, y - th - 10), (x + tw, y), (0, 0, 255), -1)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img

def format_plate_text(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned

def process_license_plate(image_path):
    img = cv2.imread(image_path)
    if img is None: 
        return "Error: Image not found", None

    # Run inference using local YOLOv8 weights
    results = model(img)
    
    detected_plates = []
    
    # Iterate through YOLOv8 detections
    for result in results:
        for box in result.boxes:
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Calculate standard bounding box measurements
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            
            # Crop out the plate area detected by YOLO
            plate_crop = img[y:y+h, x:x+w]
            if plate_crop.size == 0:
                continue
            
            
            # 1. Resize image up for clarity
            plate = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            # 2. Convert to Grayscale
            gray_crop = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

            # 3. Bilateral Filter to kill background noise while keeping text edges sharp
            denoised = cv2.bilateralFilter(gray_crop, 11, 17, 17)

            # 4. CLAHE to normalize shadows and harsh lighting variations
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast = clahe.apply(denoised)
            
            # 5. Sharpness manipulation
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            sharpened = cv2.filter2D(contrast, -1, kernel)
            
            # 6. Morphological Dilation to bolster string weight
            kernel_morph = np.ones((2, 2), np.uint8)
            final_cv_image = cv2.dilate(sharpened, kernel_morph, iterations=1)
            
            # Pass preprocessed image matrix directly into EasyOCR
            ocr_result = reader.readtext(final_cv_image, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            raw_text = "".join(ocr_result)
            text = format_plate_text(raw_text)
            
            if len(text) >= 4:
                img = draw_bounding_box(img, x, y, w, h, text)
                detected_plates.append(text)
            
    if len(detected_plates) > 0:
        final_text = ", ".join(detected_plates)
        return final_text, img
    else:
        return "No plates detected", img
