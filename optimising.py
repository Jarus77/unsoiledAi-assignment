import os
import time
import pytesseract
import multiprocessing
from pdf2image import convert_from_path
from ultralytics import YOLO
from PIL import Image

# Limit Tesseract's internal threads
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

MODEL_PATH = "/home/suraj.prasad/assignmentUnsoiled/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"
DOC_LAYNET_CLASSES = [
    "Caption","Footnote","Formula","List-item","Page-footer","Page-header",
    "Picture","Section-header","Table","Text","Title"
]
TARGET_CLASSES = {"Text", "Title"}
TESSERACT_CONFIG = "--psm 6 -l eng"

def detect_text_regions(model, page_img):
    results = model.predict(page_img, verbose=False)
    bboxes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        if DOC_LAYNET_CLASSES[cls_id] in TARGET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bboxes.append((x1, y1, x2, y2))
    return bboxes

def run_tesseract_on_bbox(args):
    img_bytes, config = args
    with Image.open(img_bytes) as crop_img:
        return pytesseract.image_to_string(crop_img, config=config)

def process_page_tesseract(page_img, bboxes, max_workers=4):
    import io
    if not bboxes:
        return ""
    tasks = []
    for (x1, y1, x2, y2) in bboxes:
        buf = io.BytesIO()
        page_img.crop((x1, y1, x2, y2)).save(buf, format="PNG")
        buf.seek(0)
        tasks.append((buf, TESSERACT_CONFIG))
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.map(run_tesseract_on_bbox, tasks)
    return "\n".join(results)

def ocr_pdf_layout(pdf_path, dpi=150, yolo_device="cuda", tesseract_workers=4):
    print(f"Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(yolo_device if yolo_device == "cuda" else "cpu")

    print("Converting PDF to images...")
    pages = convert_from_path(pdf_path, dpi=dpi)
    print(f"Total pages: {len(pages)}")

    all_text = []
    start_time = time.time()
    for i, page_img in enumerate(pages):
        t1 = time.time()
        bboxes = detect_text_regions(model, page_img)
        page_text = process_page_tesseract(page_img, bboxes, tesseract_workers)
        all_text.append(page_text)
        print(f"Page {i+1}/{len(pages)} took {time.time() - t1:.2f}s")

    print(f"Total time for {len(pages)} pages: {time.time() - start_time:.2f}s")
    return "\n".join(all_text)

if __name__ == "__main__":
    PDF_FILE = "//home/suraj.prasad/assignmentUnsoiled/attentionIsAllUNeed.pdf"
    final_text = ocr_pdf_layout(
        pdf_path=PDF_FILE,
        dpi=150,
        yolo_device="cuda",
        tesseract_workers=4
    )
    with open("output_ocr.txt", "w", encoding="utf-8") as f:
        f.write(final_text)



# python optimising.py
# Loading YOLO model: /home/suraj.prasad/assignmentUnsoiled/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt
# Converting PDF to images...
# Total pages: 15
# Page 1/15 took 5.32s
# Page 2/15 took 1.16s
# Page 3/15 took 0.77s
# Page 4/15 took 0.80s
# Page 5/15 took 0.77s
# Page 6/15 took 0.91s
# Page 7/15 took 1.00s
# Page 8/15 took 0.93s
# Page 9/15 took 0.80s
# Page 10/15 took 0.78s
# Page 11/15 took 0.09s
# Page 12/15 took 0.08s
# Page 13/15 took 0.53s
# Page 14/15 took 0.45s
# Page 15/15 took 0.12s
# Total time for 15 pages: 14.50s
