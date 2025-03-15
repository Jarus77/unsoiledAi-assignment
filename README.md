# OCR Speed Comparison

This repository compares two approaches for running OCR on PDF pages:
1. **Naive**: Uses only Tesseract (see `naive.py`).
2. **Optimized**: Uses YOLOv8 for layout segmentation (focusing on text regions) + Tesseract on the extracted regions (see `optimising.py`).

## Results

- **Naive (Tesseract only)**:  
  - Total OCR time for 15 pages: **34.46s**
- **Optimized (YOLO + Tesseract)**:  
  - Total time for 15 pages: **14.50s**  
  - Ignoring the first page’s initialization time (~5.32s), subsequent pages average ~0.66s each.

## Setup

1. Clone this repository.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure Tesseract is installed and accessible (e.g., `tesseract --version`).
4. Download the DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet weight from hugging face [DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet]

## Usage

- **Naive Approach**:  
  ```bash
  python naive.py
  ```
  This script converts PDF pages to images, then directly runs Tesseract on each page.

- **Optimized Approach**:  
  ```bash
  python optimising.py
  ```
  This script uses YOLOv8 (GPU-accelerated if available) to detect text blocks, then applies Tesseract only on those blocks to reduce total processing time.

## Notes
- Adjust file paths in the scripts to point to your PDF.
- GPU usage for YOLO can drastically improve performance. If no GPU is available, the code will default to CPU.  
- Experiment with DPI (e.g., 100–200) and Tesseract configurations for best speed and accuracy.
