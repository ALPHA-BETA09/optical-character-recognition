import easyocr
import numpy as np
from typing import List, Tuple, Dict
from src.config import LANGUAGES, USE_GPU, CONFIDENCE_THRESHOLD

class OCREngine:
    """Wrapper for the EasyOCR Reader."""
    
    def __init__(self):
        print(f"Initializing OCR Engine (Languages: {LANGUAGES}, GPU: {USE_GPU})...")
        self.reader = easyocr.Reader(LANGUAGES, gpu=USE_GPU)

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        Processes the image and extracts text.
        
        Returns a list of dictionaries containing:
        - bbox: Bounding box coordinates
        - text: Extracted string
        - confidence: Model's confidence score
        """
        print("Extracting text from image...")
        raw_results = self.reader.readtext(image)
        
        formatted_results = []
        for (bbox, text, prob) in raw_results:
            if prob >= CONFIDENCE_THRESHOLD:
                formatted_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': prob
                })
                
        return formatted_results
