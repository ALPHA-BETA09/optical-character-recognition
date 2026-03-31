import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'output')

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OCR Settings
# Add more languages if needed, e.g., ['en', 'fr', 'es']
LANGUAGES = ['en'] 
USE_GPU = True  # Set to False if you don't have a CUDA-enabled GPU
CONFIDENCE_THRESHOLD = 0.3  # Ignore text with a confidence score lower than this

# Visualization Settings
BOX_COLOR = (0, 255, 0)  # Green in BGR
TEXT_COLOR = (0, 0, 255) # Red in BGR
THICKNESS = 2
