import os
import cv2
from ultralytics import YOLO
import shutil

# Load YOLO model (replace with your model if needed)
model = YOLO('yolov10m.pt')

# Function to convert bounding box from [x1, y1, x2, y2] to [center_x, center_y, width, height]
def convert_bbox(x1, y1, x2, y2, img_width, img_height):
    # Get center x, center y, width, height
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    # Normalize the values by the image dimensions
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    return center_x, center_y, width, height

# Path to the folder containing images
image_folder = "/labeled_folder/images/"

# Paths for saving detected and non-detected images and annotations
detected_folder = "/detected_images2/"
txt_folder="/labeled_folder/labels/"
# Create output folders if they don't exist
os.makedirs(detected_folder, exist_ok=True)

# Loop through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith('.bmp'):  # Ensure it’s a BMP image file
        img_path = os.path.join(image_folder, filename)

        # Load image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        h, w, _ = img.shape  # Image height and width

        # Perform detection on the image
        results = model(img_path)

        # Check if there are any detected objects
        true_label=True
        if results[0].boxes is not None and len(results[0].boxes) > 0 :
            for box in results[0].boxes:
                    cls_id = int(box.cls.item())  # Class ID as integer
                    if cls_id>8:
                        true_label=False
        if results[0].boxes is not None and len(results[0].boxes) > 0  and true_label==True:
            # There are detections
            # Save the annotations
            txt_filename = os.path.join(txt_folder, filename.replace('.bmp', '.txt'))
            with open(txt_filename, 'w') as f:
                for box in results[0].boxes:
                    cls_id = int(box.cls.item())  # Class ID as integer
                    bbox = box.xyxy[0].cpu().numpy()  # Bounding box coordinates [x1, y1, x2, y2]

                    # Get individual coordinates
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                    # Convert to YOLO format (center_x, center_y, width, height)
                    center_x, center_y, width, height = convert_bbox(x1, y1, x2, y2, w, h)

                    # Write to the .txt file in the format: class_id center_x center_y width height
                   
                    f.write(f"{cls_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

            # Draw bounding boxes on the image
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Draw rectangle around the detected object
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add class label (optional)
                label = f"Class: {cls_id}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the image with the detections
            detected_img_with_bboxes_path = os.path.join(detected_folder, f"detected_{filename}")
            cv2.imwrite(detected_img_with_bboxes_path, img)

            print(f"Detected objects in {filename}, saved annotations to {txt_filename}, and saved image with bounding boxes to {detected_img_with_bboxes_path}")
