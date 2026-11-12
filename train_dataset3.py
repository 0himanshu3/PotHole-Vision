"""
Train and Evaluate YOLO on Dataset-3 (Pothole Detection)
======================================================
This script trains a YOLO detector on the newly converted Dataset-3
and automatically evaluates the model on both validation and test sets.
"""

from pathlib import Path
from ultralytics import YOLO
import torch

# 1. Configuration
ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = ROOT_DIR / "Dataset-3_yolo" / "data.yaml"
OUTPUT_DIR = ROOT_DIR / "runs" / "dataset3_experiment"

MODEL_NAME = "yolov8s.pt"   # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolo11s.pt, yolo11m.pt
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 15
DEVICE = 0 if torch.cuda.is_available() else "cpu"

def print_metrics(metrics_dict, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    metric_names = [
        ("Precision", "metrics/precision(B)"),
        ("Recall", "metrics/recall(B)"),
        ("mAP@0.5", "metrics/mAP50(B)"),
        ("mAP@0.5:0.95", "metrics/mAP50-95(B)"),
    ]
    for label, key in metric_names:
        val = metrics_dict.get(key)
        if val is not None:
            print(f"  {label:<18} {val:.4f} ({val*100:.2f}%)")
        else:
            print(f"  {label:<18} N/A")
    print("=" * 60 + "\n")

def main():
    print(f"Device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")
    print(f"Data config: {DATA_YAML}")
    print(f"Starting training with {MODEL_NAME} for {EPOCHS} epochs...\n")

    # 2. Load Pretrained Weights
    model = YOLO(MODEL_NAME)

    # 3. Train
    train_results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="exp_yolov8s",
        patience=PATIENCE,
        save=True,
        plots=True,
        seed=42,
        verbose=True,
    )

    # 4. Load Best Weights
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nBest model weights saved at: {best_weights}")
    best_model = YOLO(str(best_weights))

    # 5. Evaluate on Validation Set
    val_results = best_model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True
    )
    print_metrics(val_results.results_dict, "VALIDATION RESULTS — Dataset-3")

    # 6. Evaluate on Test Set
    test_results = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True
    )
    print_metrics(test_results.results_dict, "TEST RESULTS — Dataset-3")

if __name__ == "__main__":
    main()

