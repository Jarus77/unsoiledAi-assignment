import os
import time

from pdf2image import convert_from_path
import pytesseract

# For demonstration, assume there's a sample.pdf in the current directory
PDF_FILE = "/home/suraj/Downloads/researchPaper/attentionIsAllUNeed.pdf"

def ocr_pdf_naive(pdf_path):
    # Convert PDF to images (one per page)
    images = convert_from_path(pdf_path)

    # This will store the combined text from all pages
    full_text = []

    # Start the timer for the entire OCR process
    start_time = time.time()

    for i, img in enumerate(images):
        # OCR the page image
        page_start = time.time()
        text = pytesseract.image_to_string(img)
        page_end = time.time()

        # For demonstration: print out how long each page took
        print(f"Page {i+1} OCR time: {page_end - page_start:.2f} seconds")

        # Append to the full text
        full_text.append(text)

    # End the timer for the entire OCR process
    end_time = time.time()
    
    # Print total time
    print(f"Total OCR time for {len(images)} pages: {end_time - start_time:.2f} seconds")

    # Return the concatenated text for further inspection
    return "\n".join(full_text)

if __name__ == "__main__":
    extracted_text = ocr_pdf_naive(PDF_FILE)
    # Optionally, save to a text file for reference
    with open("output_naive_ocr.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)
