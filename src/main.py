from src.preprocessor import ImagePreprocessor
from src.engine import OCREngine
from src.utils import save_results

# Define the image you want to test (must be in data/input/)
IMAGE_NAME = "sample.png" 

def main():
    print(f"--- Starting OCR Process for {IMAGE_NAME} ---")
    
    try:
        # 1. Preprocess Image
        preprocessor = ImagePreprocessor(IMAGE_NAME)
        original_image = preprocessor.load_image()
        rgb_image = preprocessor.preprocess()
        
        # 2. Initialize Engine & Extract
        engine = OCREngine()
        results = engine.extract_text(rgb_image)
        
        # 3. Display raw results in console
        print("\n--- Extracted Text ---")
        for res in results:
            print(f"Text: '{res['text']}' | Confidence: {res['confidence']:.2f}")
        print("----------------------\n")
        
        # 4. Save Image and Text File
        save_results(IMAGE_NAME, original_image, results)
        
        print("--- Process Completed Successfully ---")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
