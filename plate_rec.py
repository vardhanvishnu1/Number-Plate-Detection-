import cv2
import numpy as np
import easyocr
import re

reader = easyocr.Reader(['en'], gpu=False)
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    detected_plates = []
    
    for (x, y, w, h) in plates:
        plate_crop = img[y:y+h, x:x+w]
        
        # Make the image larger for better OCR

        plate = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        #Grayscale: Remove color channels

        gray_crop = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # Bilateral Filtering: Removes noise while keeping the text edges razor-sharp

        denoised = cv2.bilateralFilter(gray_crop, 11, 17, 17)

        # CLAHE: Fixes uneven lighting and shadows dynamically

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


        contrast = clahe.apply(denoised)
        
        # Custom Sharpening Kernel: Makes the letters "pop" against the background

        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened = cv2.filter2D(contrast, -1, kernel)
        
        # Morphological Dilation: Makes the strokes of the text slightly thicker
        kernel_morph = np.ones((2, 2), np.uint8)
        final_cv_image = cv2.dilate(sharpened, kernel_morph, iterations=1)
        
        # Pass the highly processed OpenCV image to EasyOCR
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
    










