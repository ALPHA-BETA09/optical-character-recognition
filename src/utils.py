import cv2
import os
import numpy as np
from typing import List, Dict
from src.config import OUTPUT_DIR, BOX_COLOR, TEXT_COLOR, THICKNESS

def draw_boxes(image: np.ndarray, ocr_results: List[Dict]) -> np.ndarray:
    """Draws bounding boxes and text on the image."""
    annotated_image = image.copy()
    
    for result in ocr_results:
        # Extract bounding box coordinates
        top_left = tuple(map(int, result['bbox'][0]))
        bottom_right = tuple(map(int, result['bbox'][2]))
        text = result['text']
        
        # Draw Rectangle
        cv2.rectangle(annotated_image, top_left, bottom_right, BOX_COLOR, THICKNESS)
        
        # Put Text above the rectangle
        cv2.putText(
            annotated_image, 
            text, 
            (top_left[0], top_left[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            TEXT_COLOR, 
            1
        )
        
    return annotated_image

def save_results(filename: str, original_image: np.ndarray, ocr_results: List[Dict]):
    """Saves the annotated image and a text file with the extracted strings."""
    base_name = os.path.splitext(filename)[0]
    
    # 1. Save annotated image
    annotated_img = draw_boxes(original_image, ocr_results)
    image_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_annotated.jpg")
    cv2.imwrite(image_out_path, annotated_img)
    print(f"Saved annotated image to: {image_out_path}")
    
    # 2. Save text output
    text_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_text.txt")
    with open(text_out_path, 'w', encoding='utf-8') as f:
        for result in ocr_results:
            f.write(f"{result['text']} (Confidence: {result['confidence']:.2f})\n")
    print(f"Saved extracted text to: {text_out_path}")
