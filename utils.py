import cv2
import re

def draw_bounding_box(img, x1, y1, x2, y2, text=""):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    if text:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
    return img

def format_plate_text(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned