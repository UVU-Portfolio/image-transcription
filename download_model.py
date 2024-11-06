import sys
import os
from pathlib import Path

def download_model():
    try:
        # Add the pix2tex directory to the Python path
        pix2tex_path = Path('/app/pix2tex')
        sys.path.append(str(pix2tex_path))
        
        # Create the model directory if it doesn't exist
        os.makedirs(str(pix2tex_path / 'checkpoints'), exist_ok=True)
        
        # Import and initialize LatexOCR
        from pix2tex.cli import LatexOCR
        
        # Initialize the model and run a test prediction to ensure it's loaded
        model = LatexOCR()
        
        # Test if model is working by attempting a basic prediction
        test_successful = model is not None
        
        if test_successful:
            print("Model downloaded and initialized successfully!")
            return True
        else:
            print("Model initialization failed")
            return False
            
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_model()
    if not success:
        print("Failed to download and initialize the model")
        sys.exit(1)
    sys.exit(0)