import cv2
import numpy as np
import os
from src.config import INPUT_DIR

class ImagePreprocessor:
    """Handles image loading and preprocessing for OCR."""
    
    def __init__(self, filename: str):
        self.filepath = os.path.join(INPUT_DIR, filename)
        self.image = None

    def load_image(self) -> np.ndarray:
        """Loads the image from the input directory."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Image not found at: {self.filepath}")
            
        self.image = cv2.imread(self.filepath)
        return self.image

    def preprocess(self) -> np.ndarray:
        """
        Applies preprocessing to improve OCR accuracy.
        (EasyOCR is robust, so we just convert to RGB, but grayscale/thresholding 
        can be added here for noisy images).
        """
        if self.image is None:
            self.load_image()
            
        # OpenCV loads in BGR, EasyOCR expects RGB
        processed_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return processed_img
