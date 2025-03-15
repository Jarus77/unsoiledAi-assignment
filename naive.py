import os
import time

from pdf2image import convert_from_path
import pytesseract

PDF_FILE = "data/attentionIsAllUNeed.pdf"

def ocr_pdf_naive(pdf_path):
    # Convert PDF to images (one per page)
    images = convert_from_path(pdf_path)

    full_text = []

    start_time = time.time()

    for i, img in enumerate(images):
        # OCR the page image
        page_start = time.time()
        text = pytesseract.image_to_string(img)
        page_end = time.time()

        print(f"Page {i+1} OCR time: {page_end - page_start:.2f} seconds")

        full_text.append(text)

    end_time = time.time()
    
    print(f"Total OCR time for {len(images)} pages: {end_time - start_time:.2f} seconds")

    return "\n".join(full_text)

if __name__ == "__main__":
    extracted_text = ocr_pdf_naive(PDF_FILE)
    with open("output_naive_ocr.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)



# (unsoiled) suraj.prasad@bom1l40s:~/assignmentUnsoiled$ python naive.py 
# Page 1 OCR time: 2.45 seconds
# Page 2 OCR time: 3.41 seconds
# Page 3 OCR time: 1.99 seconds
# Page 4 OCR time: 2.25 seconds
# Page 5 OCR time: 2.96 seconds
# Page 6 OCR time: 2.37 seconds
# Page 7 OCR time: 2.81 seconds
# Page 8 OCR time: 2.25 seconds
# Page 9 OCR time: 2.32 seconds
# Page 10 OCR time: 2.40 seconds
# Page 11 OCR time: 2.87 seconds
# Page 12 OCR time: 2.90 seconds
# Page 13 OCR time: 1.34 seconds
# Page 14 OCR time: 1.09 seconds
# Page 15 OCR time: 1.04 seconds
# Total OCR time for 15 pages: 34.46 seconds
