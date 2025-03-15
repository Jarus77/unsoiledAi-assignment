import os
import time
import pathlib

import pytesseract
from ultralytics import YOLO
from pdf2image import convert_from_path
from PIL import Image

# -------------------------------------------------------------------------
# 1) Load the DocLayNet YOLOv8 model
#    Replace the .pt file with the path or name for your custom docLayNet weights
# -------------------------------------------------------------------------
docseg_model = YOLO("yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt")

# These are the classes the model detects (for DocLayNet). If you have a custom set, adjust accordingly.
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

# We only want to keep bounding boxes for text-based classes. 
TARGET_CLASSES = {"Text", "Title"}  
# (Optionally, you can add "Caption", "Footnote", etc. if you want to OCR those too.)

# -------------------------------------------------------------------------
# 2) Define a function to run YOLO detection on a single PIL page image
# -------------------------------------------------------------------------
def detect_text_blocks(img):
    """
    Runs the YOLO DocLayNet model on the given PIL image and returns
    bounding boxes for relevant text-like classes (Text, Title, etc.).
    """
    # docseg_model.predict returns a Results object containing bounding boxes, etc.
    results = docseg_model.predict(img, verbose=False)

    if not results:
        return []

    # results[0] is the first (and only) batch result
    text_bboxes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())  # index of the class
        class_name = DOC_LAYNET_CLASSES[cls_id]

        if class_name in TARGET_CLASSES:
            # YOLO boxes are in x1,y1,x2,y2 format
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            text_bboxes.append((class_name, x1, y1, x2, y2))
    
    return text_bboxes

# -------------------------------------------------------------------------
# 3) Define a function to crop each text block and run Tesseract
# -------------------------------------------------------------------------
def ocr_text_blocks(page_img, bboxes):
    """
    Given a PIL image (one PDF page) and bounding boxes for text areas,
    crop each region and run Tesseract OCR. Returns combined recognized text.
    """
    extracted_texts = []
    for (cls_name, x1, y1, x2, y2) in bboxes:
        # Crop the region
        crop = page_img.crop((x1, y1, x2, y2))

        # Optionally set Tesseract config if you want to speed things up further:
        # config = r'--psm 6 -l eng'  
        # text = pytesseract.image_to_string(crop, config=config)
        text = pytesseract.image_to_string(crop)  # defaults
        extracted_texts.append(text)

    # Combine text from all blocks
    return "\n".join(extracted_texts)

# -------------------------------------------------------------------------
# 4) Main pipeline to process an entire PDF (single-threaded example)
# -------------------------------------------------------------------------
def process_pdf_with_doclaynet(pdf_path, dpi=200):
    """
    - Convert the PDF into a list of PIL images at a chosen DPI.
    - Use the docLayNet YOLOv8 model to find text-based bounding boxes.
    - Run Tesseract on each bounding box.
    - Return the full extracted text.
    """
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=dpi)
    
    all_pages_text = []
    start_time = time.time()
    
    for page_idx, page_img in enumerate(pages, start=1):
        page_start = time.time()
        # 1) Detect text blocks
        bboxes = detect_text_blocks(page_img)

        # 2) OCR those text blocks
        page_text = ocr_text_blocks(page_img, bboxes)
        all_pages_text.append(page_text)

        page_end = time.time()
        print(f"[Page {page_idx}] processed in {page_end - page_start:.2f} sec")

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds for {len(pages)} pages")

    return "\n".join(all_pages_text)

# -------------------------------------------------------------------------
# 5) Run the pipeline
# -------------------------------------------------------------------------
if __name__ == "__main__":
    pdf_file = "/home/suraj/Downloads/researchPaper/attentionIsAllUNeed.pdf"  # Replace with your own file
    output_text = process_pdf_with_doclaynet(pdf_file, dpi=200)

    # Save to disk or use as needed
    with open("doclaynet_ocr_output.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
