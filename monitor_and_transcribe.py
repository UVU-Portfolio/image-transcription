import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
from docx import Document
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MONITORED_DIR = '/mnt/d/OCR-stuff/midterm-prep'
OUTPUT_DIR = '/mnt/d/OCR-stuff/midterm-prep-output'
PIX2TEX_PATH = '/app/pix2tex/run.py'

def get_batch_doc_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f'batch_transcriptions_{timestamp}.docx')

def transcribe_image(image_path):
    try:
        logging.info(f"Attempting to transcribe image: {image_path}")
        command = f'python3 {PIX2TEX_PATH} --image {image_path}'
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        logging.info(f"Transcription result: {result.stdout.strip()}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Error transcribing {image_path}: {e}")
        logging.error(f"Command output: {e.output}")
        return None

class ImageHandler(FileSystemEventHandler):
    def __init__(self, doc_path):
        self.doc_path = doc_path
        self.create_new_doc()

    def create_new_doc(self):
        try:
            doc = Document()
            doc.add_heading('Batch OCR Transcriptions', level=0)
            doc.save(self.doc_path)
            logging.info(f"Created new batch document: {self.doc_path}")
        except Exception as e:
            logging.error(f"Error creating document: {e}")

    def process_image(self, image_path):
        logging.info(f"Processing image: {image_path}")
        transcription = transcribe_image(image_path)
        if transcription:
            self.append_to_doc(image_path, transcription)

    def on_created(self, event):
        if event.is_directory:
            return
        self.process_image(event.src_path)

    def append_to_doc(self, image_path, transcription):
        try:
            doc = Document(self.doc_path)
            doc.add_heading(f'Transcription for {os.path.basename(image_path)}', level=1)
            doc.add_paragraph(transcription)
            doc.save(self.doc_path)
            logging.info(f"Appended transcription for {os.path.basename(image_path)} to {self.doc_path}")
        except Exception as e:
            logging.error(f"Error appending to document: {e}")

def process_existing_files(handler):
    for filename in os.listdir(MONITORED_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(MONITORED_DIR, filename)
            handler.process_image(file_path)

def run_test():
    logging.info("Running test function")
    test_image_path = os.path.join(MONITORED_DIR, 'IMG_8805.jpeg')  # Using the first image in your directory
    if os.path.exists(test_image_path):
        transcription = transcribe_image(test_image_path)
        if transcription:
            logging.info(f"Test transcription successful: {transcription}")
            test_doc_path = os.path.join(OUTPUT_DIR, 'test_transcription.docx')
            try:
                doc = Document()
                doc.add_paragraph(transcription)
                doc.save(test_doc_path)
                logging.info(f"Test document saved successfully: {test_doc_path}")
            except Exception as e:
                logging.error(f"Error saving test document: {e}")
        else:
            logging.error("Test transcription failed")
    else:
        logging.error(f"Test image not found: {test_image_path}")

def main():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info(f"Ensured output directory exists: {OUTPUT_DIR}")
    except Exception as e:
        logging.error(f"Error creating output directory: {e}")
        return

    run_test()

    batch_doc_path = get_batch_doc_path()
    event_handler = ImageHandler(batch_doc_path)
    
    process_existing_files(event_handler)

    observer = Observer()
    observer.schedule(event_handler, path=MONITORED_DIR, recursive=False)
    observer.start()

    logging.info(f"Monitoring directory: {MONITORED_DIR}")
    logging.info(f"All transcriptions will be appended to: {batch_doc_path}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()