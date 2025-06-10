# YOLOv10 BMP Annotator

This script detects objects in `.bmp` images using a YOLOv10 model, saves YOLO-format annotations, and visualizes detections.

## Features
- Only keeps objects with class ID ≤ 8
- Saves YOLO `.txt` annotations
- Draws and saves detection visualizations

## How to Run
Edit the image folder and output paths in the script, then run:

```bash
python detect_and_annotate.py
