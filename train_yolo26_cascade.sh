#!/bin/bash

# YOLO26 Training Script for Cascade Pipeline (Stage 1: Detection)
# Uses latest YOLO26 architecture
# This trains a single super-class detector for "plankton"
# Stage 2 (ArcFace) will handle species identification

set -e  # Exit on error

echo "========================================"
echo "YOLO26 CASCADE TRAINING - STAGE 1"
echo "========================================"

# Configuration
MODEL="yolo26l"  # YOLO26-Large for better performance
DATASET_DIR="yolo_superclass_dataset"
DATASET_YAML="${DATASET_DIR}/dataset.yaml"
PROJECT_NAME="yolo26_cascade_training"
EPOCHS=100
BATCH_SIZE=2  # Small batch for full resolution (reduce to 1 if OOM)
IMG_SIZE=1920  # Full native resolution - no resizing!
DEVICE=0  # GPU 0, use "cpu" for CPU training

# Detection-specific parameters (high recall)
CONF_THRESHOLD=0.15  # Low confidence for high recall
IOU_THRESHOLD=0.45   # IoU for NMS (YOLO26 has improved NMS-free inference)
MAX_DET=300          # Max detections per image

echo ""
echo "Configuration:"
echo "  Model: ${MODEL} (YOLO26 - Latest)"
echo "  Dataset: ${DATASET_YAML}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Image Size: ${IMG_SIZE}"
echo "  Confidence Threshold: ${CONF_THRESHOLD}"
echo "  Latest Architecture: Yes"
echo ""

# Step 1: Check ultralytics version
echo "Checking Ultralytics version..."
python3 -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"
echo ""
echo "Note: YOLO26 requires ultralytics>=8.4.0"
echo "      If version is too old, run: pip install -U ultralytics"
echo ""

# Step 2: Create super-class dataset if not exists
if [ ! -f "${DATASET_YAML}" ]; then
    echo "Dataset not found. Creating super-class dataset..."
    python3 create_superclass_yolo_dataset.py
    echo "Dataset created!"
else
    echo "Dataset already exists at ${DATASET_YAML}"
fi

# Step 3: Check if dataset YAML exists
if [ ! -f "${DATASET_YAML}" ]; then
    echo "ERROR: Dataset YAML not found at ${DATASET_YAML}"
    echo "Please run: python3 create_superclass_yolo_dataset.py"
    exit 1
fi

echo ""
echo "========================================"
echo "Starting YOLO26 Training..."
echo "========================================"
echo ""

# Step 4: Train YOLO26 model
yolo detect train \
    data="${DATASET_YAML}" \
    model="${MODEL}.pt" \
    epochs=${EPOCHS} \
    imgsz=${IMG_SIZE} \
    batch=${BATCH_SIZE} \
    device=${DEVICE} \
    project="${PROJECT_NAME}" \
    name="yolo26_superclass_plankton" \
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
    copy_paste=0.0 \
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
BEST_MODEL="${PROJECT_NAME}/yolo26_superclass_plankton/weights/best.pt"

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

    # Step 7: Export model (optional)
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
    echo "Results directory: ${PROJECT_NAME}/yolo26_superclass_plankton/"
    echo ""
    echo "YOLO11 Advantages:"
    echo "  ✓ Latest architecture (Sep 2024)"
    echo "  ✓ Better small object detection than YOLOv10"
    echo "  ✓ Improved training efficiency"
    echo "  ✓ Optimized for edge deployment"
    echo ""
    echo "Next steps:"
    echo "  1. Check validation results in ${PROJECT_NAME}/yolo26_superclass_plankton/"
    echo "  2. Compare with YOLOv10 results (mAP, recall, speed)"
    echo "  3. If better, use YOLO11 for cascade inference"
    echo "  4. Or continue with: python3 cascade_inference.py --yolo ${BEST_MODEL}"
    echo ""
else
    echo "ERROR: Best model not found at ${BEST_MODEL}"
    exit 1
fi
