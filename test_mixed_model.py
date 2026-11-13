"""
Test Environment for Mixed-Trained Pothole Detection Model
==========================================================
Upload or specify an image, and this script detects potholes by drawing
clean bounding boxes around them (without confidence scores).

Can be run:
1. Locally with Python: python test_mixed_model.py --image path/to/image.jpg
2. Or imported into Google Colab / Jupyter notebook.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. Path to your best mixed model weights
# In Colab: Path("/content/drive/MyDrive/pothole_project/runs/exp_mixed/weights/best.pt")
# Locally: Adjust to where you saved your weights
MODEL_PATH = Path("runs/exp_mixed/weights/best.pt")

def detect_potholes(image_input, model=None, conf_threshold=0.25, line_thickness=3, box_color=(0, 255, 0), show_label=True):
    """
    Detects potholes in an image and draws clean rectangles without confidence numbers.
    
    Args:
        image_input: image path (str or Path) or numpy BGR array from cv2.imread.
        model: loaded YOLO model instance.
        conf_threshold: confidence cutoff to filter out weak detections.
        line_thickness: thickness of the bounding box lines in pixels.
        box_color: BGR tuple for box color (default: (0, 255, 0) bright green).
        show_label: if True, writes 'pothole' above the box without any score.
    """
    if model is None:
        model = YOLO(str(MODEL_PATH))
    
    # Load image if a path is provided
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Could not load image from: {image_input}")
    else:
        img = image_input.copy()

    # Run inference
    results = model.predict(img, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes

    annotated = img.copy()
    count = 0

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Extract coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # 1. Draw the clean bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, line_thickness)
            count += 1

            # 2. Optional: Draw 'pothole' badge with NO confidence score
            if show_label:
                label_text = "pothole"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                # Calculate text size for background banner
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                
                # Draw small background pill behind label for readability
                badge_y1 = max(0, y1 - text_h - 8)
                badge_y2 = y1
                cv2.rectangle(annotated, (x1, badge_y1), (x1 + text_w + 6, badge_y2), box_color, -1)
                
                # Draw text in contrasting black
                cv2.putText(annotated, label_text, (x1 + 3, y1 - 4), font, font_scale, (0, 0, 0), font_thickness)

    return annotated, count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pothole Detection Testing Environment (No Confidence Scores)")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--weights", type=str, default=str(MODEL_PATH), help="Path to best.pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    args = parser.parse_args()

    if args.image:
        model = YOLO(args.weights)
        output_img, detected_count = detect_potholes(args.image, model=model, conf_threshold=args.conf)
        print(f"Done! Detected {detected_count} pothole(s).")
        
        # Save output
        out_name = f"detected_{Path(args.image).name}"
        cv2.imwrite(out_name, output_img)
        print(f"Saved annotated image to: {out_name}")
    else:
        print("Please provide an image: python test_mixed_model.py --image your_image.jpg")
