#!/bin/bash

# YOLO11 End-to-End Training Script (Detection + Classification)
# Uses YOLO11 for both detection AND classification (39 classes)
# For comparison against 2-stage cascade approach

set -e  # Exit on error

echo "========================================"
echo "YOLO11 END-TO-END TRAINING"
echo "========================================"
echo ""
echo "This trains a SINGLE model for:"
echo "  1. Detection (bounding boxes)"
echo "  2. Classification (39 plankton species)"
echo ""
echo "Compare with Cascade Approach:"
echo "  - Cascade: YOLO (1 class) → ArcFace (39 classes)"
echo "  - End-to-End: YOLO11 (39 classes)"
echo ""

# Configuration
MODEL="yolo11l"  # YOLO11-Large
DATASET_DIR="yolo_multiclass_dataset"
DATASET_YAML="${DATASET_DIR}/dataset.yaml"
PROJECT_NAME="yolo26_multiclass_training"
EPOCHS=100
BATCH_SIZE=2  # Adjust based on GPU memory
IMG_SIZE=1920  # Full native resolution
DEVICE=0

# Detection parameters (balanced for all classes)
CONF_THRESHOLD=0.25  # Higher than cascade (more classes)
IOU_THRESHOLD=0.45
MAX_DET=300

echo ""
echo "Configuration:"
echo "  Model: ${MODEL} (YOLO11 - Sep 2024)"
echo "  Classes: 39 (all plankton species)"
echo "  Dataset: ${DATASET_YAML}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Image Size: ${IMG_SIZE}"
echo "  Architecture: End-to-End Detection + Classification"
echo ""

# Step 1: Check ultralytics version
echo "Checking Ultralytics version..."
python3 -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"
echo ""
echo "Note: YOLO11 requires ultralytics>=8.3.0"
echo ""

# Step 2: Create multi-class dataset if not exists
if [ ! -f "${DATASET_YAML}" ]; then
    echo "Multi-class dataset not found. Creating..."
    python3 create_multiclass_yolo_dataset.py
    echo "Dataset created!"
else
    echo "Dataset already exists at ${DATASET_YAML}"
fi

# Step 3: Check if dataset YAML exists
if [ ! -f "${DATASET_YAML}" ]; then
    echo "ERROR: Dataset YAML not found at ${DATASET_YAML}"
    echo "Please run: python3 create_multiclass_yolo_dataset.py"
    exit 1
fi

echo ""
echo "========================================"
echo "Starting YOLO11 Multi-Class Training..."
echo "========================================"
echo ""

# Step 4: Train YOLO11 model with all 39 classes
yolo detect train \
    data="${DATASET_YAML}" \
    model="${MODEL}.pt" \
    epochs=${EPOCHS} \
    imgsz=${IMG_SIZE} \
    batch=${BATCH_SIZE} \
    device=${DEVICE} \
    project="${PROJECT_NAME}" \
    name="yolo26_multiclass_39species" \
    exist_ok=True \
    pretrained=True \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    momentum=0.9 \
    weight_decay=0.0005 \
    warmup_epochs=3.0 \
    warmup_momentum=0.8 \
    warmup_bias_lr=0.1 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    conf=${CONF_THRESHOLD} \
    iou=${IOU_THRESHOLD} \
    max_det=${MAX_DET} \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=45 \
    translate=0.1 \
    scale=0.5 \
    flipud=0.5 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.0 \
    copy_paste=0.3 \
    rect=True \
    patience=20 \
    save=True \
    val=True \
    plots=True \
    cache=False \
    amp=True \
    workers=8 \
    verbose=True

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"

# Step 5: Find best model path
BEST_MODEL="${PROJECT_NAME}/yolo26_multiclass_39species/weights/best.pt"

if [ -f "${BEST_MODEL}" ]; then
    echo ""
    echo "Best model saved at: ${BEST_MODEL}"

    # Step 6: Validate model
    echo ""
    echo "Running validation..."
    yolo detect val \
        model="${BEST_MODEL}" \
        data="${DATASET_YAML}" \
        imgsz=${IMG_SIZE} \
        conf=${CONF_THRESHOLD} \
        iou=${IOU_THRESHOLD} \
        max_det=${MAX_DET} \
        device=${DEVICE}

    # Step 7: Generate per-class metrics
    echo ""
    echo "Generating per-class analysis..."
    python3 << 'PYTHON_SCRIPT'
import json
import os

# Read validation results
results_dir = "yolo26_multiclass_training/yolo26_multiclass_39species"
print(f"\nResults saved to: {results_dir}")
print("\nCheck the following files for detailed metrics:")
print(f"  - {results_dir}/confusion_matrix.png")
print(f"  - {results_dir}/results.csv")
print(f"  - {results_dir}/PR_curve.png")
print(f"  - {results_dir}/F1_curve.png")
PYTHON_SCRIPT

    # Step 8: Export model
    echo ""
    echo "Exporting model to ONNX format..."
    yolo export \
        model="${BEST_MODEL}" \
        format=onnx \
        simplify=True \
        dynamic=True \
        imgsz=${IMG_SIZE}

    echo ""
    echo "========================================"
    echo "All Done!"
    echo "========================================"
    echo ""
    echo "Model location: ${BEST_MODEL}"
    echo "Results directory: ${PROJECT_NAME}/yolo26_multiclass_39species/"
    echo ""
    echo "End-to-End YOLO11 Advantages:"
    echo "  ✓ Single model (simpler deployment)"
    echo "  ✓ Faster inference (no 2-stage processing)"
    echo "  ✓ Direct class predictions with bboxes"
    echo "  ✓ Latest architecture (Sep 2024)"
    echo ""
    echo "Next steps for comparison:"
    echo "  1. Compare detection metrics with cascade YOLO:"
    echo "     - mAP@0.5, mAP@0.5:0.95"
    echo "     - Per-class AP"
    echo "     - Recall for small objects (Chlorella sp)"
    echo ""
    echo "  2. Compare classification metrics with cascade (YOLO+ArcFace):"
    echo "     - Overall accuracy (Cascade: 98.21%)"
    echo "     - Per-class precision/recall"
    echo "     - Confusion matrix"
    echo ""
    echo "  3. Compare practical metrics:"
    echo "     - Inference speed (FPS)"
    echo "     - Memory usage"
    echo "     - Deployment complexity"
    echo ""
    echo "  4. Run comparison script:"
    echo "     python3 compare_yolo11_vs_cascade.py"
    echo ""
else
    echo "ERROR: Best model not found at ${BEST_MODEL}"
    exit 1
fi
