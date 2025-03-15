import os
import time
import pathlib

import pytesseract
from ultralytics import YOLO
from pdf2image import convert_from_path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------
# GLOBALS / CONSTANTS
# ------------------------------------------------------------

# Path to your YOLOv8 DocLayNet model
MODEL_PATH = "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"

DOC_LAYNET_CLASSES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title"
]
TARGET_CLASSES = {"Text", "Title"}  # We only want text-related boxes

# To speed up Tesseract for English text blocks
# - psm 6 is "Assume a single uniform block of text."
# - l eng means only English language
TESSERACT_CONFIG = r"--psm 6 -l eng"


# ------------------------------------------------------------
# 1) Single-page processing function
# ------------------------------------------------------------
def process_single_page(page_idx, page_img):
    """
    1) Load YOLO model (DocLayNet) - each process will do this once.
    2) Detect text bounding boxes on the page image.
    3) Crop each box and run Tesseract.
    4) Return recognized text + page index.
    """
    # Initialize the YOLO model inside the worker process
    # so each process can run detection independently.
    docseg_model = YOLO(MODEL_PATH)

    # Run detection
    results = docseg_model.predict(page_img, verbose=False)

    if not results:
        return page_idx, ""

    # Extract bounding boxes for text classes
    text_bboxes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())  # index of the class
        class_name = DOC_LAYNET_CLASSES[cls_id]

        if class_name in TARGET_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            text_bboxes.append((x1, y1, x2, y2))

    # OCR the text regions
    extracted_texts = []
    for (x1, y1, x2, y2) in text_bboxes:
        crop = page_img.crop((x1, y1, x2, y2))
        # Use a custom Tesseract config for speed + English
        text = pytesseract.image_to_string(crop, config=TESSERACT_CONFIG)
        extracted_texts.append(text)

    page_text = "\n".join(extracted_texts)
    return page_idx, page_text


# ------------------------------------------------------------
# 2) Main pipeline with parallel processing
# ------------------------------------------------------------
def process_pdf_parallel(pdf_path, dpi=200, max_workers=4):
    """
    - Convert PDF to images
    - Run YOLO detection + Tesseract in parallel for each page
    - Return combined text
    """
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=dpi)
    num_pages = len(pages)

    start_time = time.time()
    results_text = [None] * num_pages  # to store text in correct page order

    # Use ProcessPoolExecutor for parallel CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs
        futures = []
        for idx, page_img in enumerate(pages):
            future = executor.submit(process_single_page, idx, page_img)
            futures.append(future)

        # Collect results
        for future in as_completed(futures):
            page_idx, page_text = future.result()
            results_text[page_idx] = page_text

    total_time = time.time() - start_time
    print(f"Processed {num_pages} pages in {total_time:.2f} seconds (parallel)")

    return "\n".join(results_text)


# ------------------------------------------------------------
# 3) Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    pdf_file = "/home/suraj/Downloads/researchPaper/attentionIsAllUNeed.pdf"
    
    # Adjust these parameters as you see fit:
    # - dpi=150 or 200 
    # - max_workers=4 (or 8 if you have 8 cores)
    final_text = process_pdf_parallel(pdf_file, dpi=150, max_workers=4)

    with open("doclaynet_ocr_output_parallel.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
