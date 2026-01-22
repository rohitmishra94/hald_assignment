# Fine-Grained Plankton Identification: 2-Stage Cascade System

## Overview

This project implements a **high-precision fine-grained object identification system** for plankton classification using a **2-stage cascade approach** to handle:
- **High intra-class variance**: Same species looks different due to rotation, lighting, morphology
- **High inter-class similarity**: Different species look nearly identical

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STAGE 1: DETECTION                       │
│                                                              │
│  ┌────────────┐        ┌──────────────┐                    │
│  │   Image    │   →    │     YOLO     │   →  Bounding Boxes│
│  └────────────┘        │ (Super-Class)│                    │
│                        └──────────────┘                     │
│                                                              │
│  Goal: Detect ALL plankton objects (High Recall)            │
│  Single super-class: "plankton"                              │
│  Low confidence threshold (0.15) to catch all objects        │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Cropped Object Images
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 2: IDENTIFICATION                    │
│                                                              │
│  ┌────────────┐        ┌──────────────┐                    │
│  │  Cropped   │   →    │   ArcFace    │   →  Species ID   │
│  │   Object   │        │ (Metric      │      + Confidence  │
│  └────────────┘        │  Learning)   │                    │
│                        └──────────────┘                     │
│                              ↓                               │
│                     Cosine Similarity                        │
│                              ↓                               │
│                     Class Prototypes                         │
│                                                              │
│  Goal: Identify species using deep metric learning          │
│  Sub-Center ArcFace (K=3) for intra-class variance          │
│  Hard labels + Angular margin for inter-class separation    │
└─────────────────────────────────────────────────────────────┘
```

## Key Technical Decisions

### Stage 1: YOLO Detection
- **Model**: YOLOv10 (nano/small)
- **Classes**: Single super-class "plankton"
- **Strategy**: Optimize for **HIGH RECALL** (detect all objects)
- **Confidence Threshold**: Low (0.15) - false positives acceptable
- **Rationale**: Let ArcFace handle false positives with high confidence threshold

### Stage 2: ArcFace Identification
- **Loss Function**: **Sub-Center ArcFace** (K=3 sub-centers per class)
  - Handles intra-class variance by allowing multiple clusters per class
  - Standard ArcFace would struggle with morphological variations
- **Backbone**: ResNet-18/50 (FC layer removed, 512-dim embeddings)
- **Training**:
  - Hard labels (no label smoothing) - maintains angular margins
  - Aggressive margin (m=0.5) for inter-class separation
  - Scale (s=64.0) for stable gradients
- **Inference**:
  - Discard ArcFace Head (no longer needed)
  - Extract L2-normalized embeddings
  - Match against class prototypes using cosine similarity
  - Confidence threshold (0.6) for final predictions

## Directory Structure

```
hald_assignment/
├── data_audit/                        # Task 1: Data analysis
│   ├── data_analysis.ipynb
│   ├── intra_class_analysis.ipynb
│   └── data_audit/
│       ├── class_distribution.png
│       ├── bbox_distribution.png
│       └── interclass_umap.png
│
├── evaluation/                        # Task 2: Model code
│   ├── model_arc.py                   # ArcFace models
│   ├── train_arc.py                   # ArcFace training
│   ├── inference_arc.py               # ArcFace inference
│   ├── train_yolo.py                  # YOLO training
│   └── inference_yolo.py              # YOLO inference
│
├── prepare_arcface_dataset.py         # Convert COCO → ArcFace dataset
├── create_superclass_yolo_dataset.py  # Create super-class YOLO dataset
├── generate_prototypes.py             # Generate class prototypes
├── cascade_inference.py               # Unified cascade pipeline
├── split_dataset.py                   # Smart train/test split
│
├── annotation.coco.json               # Original COCO annotations
├── images/                            # Original images
│
├── yolo_superclass_dataset/           # Generated YOLO dataset
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── dataset.yaml
│
├── arcface_dataset/                   # Generated ArcFace dataset
│   ├── train/class1/, class2/, ...
│   ├── test/class1/, class2/, ...
│   ├── dataset_info.json
│   └── class_mapping.json
│
└── CASCADE_PIPELINE_README.md         # This file
```

## Pipeline Workflow

### Step 1: Prepare Datasets

#### 1.1 Create YOLO Super-Class Dataset
```bash
python create_superclass_yolo_dataset.py
```
**Output**:
- `yolo_superclass_dataset/` with single "plankton" class
- All species → single class for detection only

#### 1.2 Create ArcFace Dataset
```bash
python prepare_arcface_dataset.py
```
**Output**:
- `arcface_dataset/train/` - Cropped 128x128 objects by class
- `arcface_dataset/test/` - Test set
- Rare classes augmented automatically

### Step 2: Train Models

#### 2.1 Train YOLO Detector (Stage 1)
```bash
cd evaluation
python train_yolo.py --data ../yolo_superclass_dataset/dataset.yaml \
                     --epochs 100 \
                     --batch 16 \
                     --conf 0.15
```

**Training Parameters**:
- Low confidence threshold for high recall
- Standard augmentation (mosaic, mixup)
- Focus on **not missing** any objects

**Expected Output**:
- `yolo_superclass_dataset/runs/detect/train/weights/best.pt`
- mAP@0.5 > 0.8 (detection quality)
- High recall (>0.95) even if precision suffers

#### 2.2 Train ArcFace Identifier (Stage 2)
```bash
cd evaluation
python train_arc.py --dataset ../arcface_dataset \
                    --epochs 50 \
                    --batch 64 \
                    --embedding 512
```

**Training Parameters**:
- Sub-Center ArcFace with K=3
- Angular margin m=0.5
- Scale s=64.0
- No label smoothing (hard labels)

**Expected Output**:
- `arcface_models/best_model.pth`
- Top-1 accuracy > 90%
- Good inter-class separation

### Step 3: Generate Class Prototypes

```bash
python generate_prototypes.py \
    --model arcface_models/best_model.pth \
    --dataset arcface_dataset \
    --output arcface_models/class_prototypes.pth
```

**What it does**:
- Extracts embeddings for all training samples
- Averages embeddings per class → class prototype
- L2-normalizes prototypes
- Saves for inference

**Output**:
- `arcface_models/class_prototypes.pth`
- `arcface_models/prototype_statistics.txt`
- `arcface_models/embedding_visualization_umap.png`

### Step 4: Run Cascade Inference

#### Single Image
```bash
python cascade_inference.py \
    --image path/to/image.jpg \
    --yolo yolo_superclass_dataset/runs/detect/train/weights/best.pt \
    --arcface arcface_models/best_model.pth \
    --prototypes arcface_models/class_prototypes.pth \
    --mapping arcface_dataset/class_mapping.json \
    --yolo-conf 0.15 \
    --arcface-conf 0.6
```

#### Batch Processing
```bash
python cascade_inference.py \
    --batch path/to/images/ \
    --output cascade_results/ \
    --yolo-conf 0.15 \
    --arcface-conf 0.6
```

**Output**:
- Annotated images with species labels
- `cascade_results/report.txt` with per-class counts
- Confidence scores for each detection

## Performance Considerations

### Stage 1 (YOLO)
- **Speed**: ~100 FPS on GPU (YOLOv10-nano)
- **Recall**: Aim for >95% (don't miss objects)
- **Precision**: Less critical (Stage 2 filters false positives)

### Stage 2 (ArcFace)
- **Speed**: ~500 FPS for embedding extraction (batch)
- **Accuracy**: Top-1 >90%, Top-5 >98%
- **Confidence**: Only trust predictions >0.6 similarity

### End-to-End Pipeline
- **Combined Speed**: ~50-100 FPS (depends on object density)
- **Accuracy**: 85-95% (high confidence predictions only)
- **Robustness**: Handles rotation, lighting, scale variations

## Key Advantages of This Approach

1. **Decoupling Detection & Identification**
   - YOLO: Simple single-class detection (fast, high recall)
   - ArcFace: Complex fine-grained identification (accurate)

2. **Sub-Center ArcFace**
   - K=3 allows 3 "modes" per class
   - Handles morphological variations naturally
   - Better than standard ArcFace for biological objects

3. **Hard Labels + Angular Margin**
   - No label smoothing → sharp decision boundaries
   - Critical for separating similar species
   - m=0.5 margin forces inter-class separation

4. **Prototype-Based Inference**
   - No need for classification head at inference
   - Easy to add new classes (just add prototype)
   - Interpretable (cosine similarity)

5. **Confidence Calibration**
   - YOLO confidence: Is it an object?
   - ArcFace confidence: Which species?
   - Reject low-confidence predictions

## Troubleshooting

### Issue: YOLO misses small objects
**Solution**:
- Lower confidence threshold further (0.10)
- Train with smaller anchor boxes
- Use larger image size (1024x1024)

### Issue: ArcFace confuses similar classes
**Solution**:
- Increase angular margin (m=0.6 or 0.7)
- Add more training data for confused classes
- Increase K (more sub-centers per class)

### Issue: Low overall accuracy
**Solution**:
- Check YOLO recall (should be >95%)
- Check ArcFace top-5 accuracy (should be >98%)
- Visualize embeddings (UMAP) to check separation
- Augment rare/confused classes

### Issue: False positives from YOLO
**Solution**:
- Raise ArcFace confidence threshold
- Add "background" class to ArcFace
- Filter by bbox size/aspect ratio

## Evaluation Metrics

### Detection Metrics (Stage 1)
- mAP@0.5, mAP@0.5:0.95
- **Recall** (most important)
- Precision (less critical)

### Identification Metrics (Stage 2)
- Top-1 accuracy
- Top-5 accuracy
- Per-class precision/recall
- Confusion matrix

### Counting Metrics (End-to-End)
- MAE (Mean Absolute Error)
- Per-class count accuracy
- Count accuracy by density (sparse/medium/dense)

### Cascade Metrics
- End-to-end accuracy
- Confident prediction rate
- Processing speed (FPS)

## References

- **ArcFace**: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **Sub-Center ArcFace**: [Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)
- **YOLOv10**: [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)

## Citation

```bibtex
@article{plankton_cascade_2025,
  title={Fine-Grained Plankton Identification using 2-Stage Cascade with Sub-Center ArcFace},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.

---

**Generated with Claude Code** 🤖
