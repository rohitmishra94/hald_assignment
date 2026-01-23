# Fine-Grained Plankton Identification System

A high-precision 2-stage cascade pipeline for plankton species identification from microscopy images, addressing the challenges of high intra-class variance and inter-class similarity in fine-grained biological classification.

## 🎯 Overview

This project implements a production-ready plankton identification system using a **2-stage cascade architecture**:

1. **Stage 1 (YOLO Detection)**: Detect ALL plankton objects with high recall
2. **Stage 2 (ArcFace Classification)**: Identify species using deep metric learning

### Key Challenges Addressed
- **High intra-class variance**: Same species looks different due to rotation, lighting, morphology
- **High inter-class similarity**: Different species look nearly identical
- **Class imbalance**: Rare species with few samples
- **Fine-grained features**: Subtle differences between species

## 📊 Performance

### Stage 1: YOLO Detection
- **Model**: YOLOv10-Large (single "plankton" super-class)
- **mAP@0.5**: 91.6%
- **Recall**: 85.9%
- **Resolution**: 1920×1080 (full native resolution)

### Stage 2: ArcFace Identification
- **Model**: ResNet50 + Sub-Center ArcFace (K=5)
- **Top-1 Accuracy**: 94.83%
- **Top-5 Accuracy**: 98.28%
- **F1-Macro**: 0.8391 (all classes treated equally)
- **F1-Weighted**: 0.9484 (weighted by class frequency)
- **Classes**: 39 plankton species

### End-to-End Pipeline
- **Combined Accuracy**: ~85-90% (high confidence predictions only)
- **Speed**: ~50-100 FPS (depends on object density)
- **Confidence Thresholds**: YOLO=0.15, ArcFace=0.6

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/rohitmishra94/hald_assignment.git
cd hald_assignment

# Install dependencies
pip install -r requirements.txt

# Install YOLOv10
pip install ultralytics
```

### 2. Dataset Preparation

```bash
# Create YOLO super-class dataset (single "plankton" class)
python create_superclass_yolo_dataset.py

# Create ArcFace dataset (cropped 128×128 objects by species)
python prepare_arcface_dataset.py
```

**Output**:
- `yolo_superclass_dataset/` - YOLO training data
- `arcface_dataset/` - ArcFace training data (train/test splits)

### 3. Training

#### Stage 1: Train YOLO Detector
```bash
bash train_yolo_cascade.sh
```

**Configuration**:
- Model: YOLOv10-Large
- Resolution: 1920×1080
- Epochs: 100
- Batch size: 2
- Confidence: 0.15 (high recall)

#### Stage 2: Train ArcFace Identifier
```bash
cd evaluation
python train_arc.py \
    --dataset ../arcface_dataset \
    --backbone resnet50 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001
```

**Configuration**:
- Backbone: ResNet50
- Optimizer: AdamW (lr=0.0001)
- Scheduler: CosineAnnealingLR
- Sub-Center ArcFace: K=5, s=30, m=0.35
- Augmentation: Rotation, affine, color jitter, random erasing

**Training outputs** (saved to `arcface_models/`):
- `best_model.pth` - Best model (based on F1-Macro)
- `classification_report.txt` - Per-class metrics
- `confusion_matrix.png` - Species confusion heatmap
- `training_history.png` - Loss/accuracy/F1 curves

### 4. Generate Class Prototypes

```bash
python generate_prototypes.py \
    --model arcface_models/best_model.pth \
    --dataset arcface_dataset \
    --backbone resnet50 \
    --output arcface_models/class_prototypes.pth
```

**Output**:
- `class_prototypes.pth` - Averaged embeddings per class
- `embedding_visualization_umap.png` - UMAP visualization
- `prototype_statistics.txt` - Inter-class similarity analysis

### 5. Run Inference

#### Single Image
```bash
python cascade_inference.py \
    --image path/to/image.jpg \
    --yolo yolo_cascade_training/yolo_superclass_plankton/weights/best.pt \
    --arcface arcface_models/best_model.pth \
    --prototypes arcface_models/class_prototypes.pth \
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

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STAGE 1: DETECTION                       │
│                                                              │
│  ┌────────────┐        ┌──────────────┐                    │
│  │   Image    │   →    │   YOLOv10    │   →  Bounding Boxes│
│  │ 1920×1080  │        │ (Super-Class)│                    │
│  └────────────┘        └──────────────┘                     │
│                                                              │
│  Goal: Detect ALL plankton objects (High Recall)            │
│  Confidence: 0.15 (low threshold to catch everything)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Cropped Object Images
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 2: IDENTIFICATION                    │
│                                                              │
│  ┌────────────┐        ┌──────────────┐                    │
│  │  Cropped   │   →    │   ResNet50   │   →  512-dim      │
│  │ 128×128    │        │   Backbone   │      Embedding     │
│  └────────────┘        └──────────────┘                     │
│                              ↓                               │
│                     L2 Normalization                         │
│                              ↓                               │
│                    Cosine Similarity                         │
│                              ↓                               │
│                     Class Prototypes                         │
│                              ↓                               │
│                   Species ID + Confidence                    │
│                                                              │
│  Goal: Identify species using metric learning               │
│  Sub-Center ArcFace (K=5) for intra-class variance          │
│  Confidence: 0.6 (only trust high similarity predictions)   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
hald_assignment/
├── README.md                           # This file
├── CASCADE_PIPELINE_README.md          # Detailed technical documentation
│
├── data_audit/                         # Data analysis and visualization
│   ├── data_analysis.ipynb             # Class distribution, bbox analysis
│   ├── intra_class_analysis.ipynb      # Intra-class variance study
│   └── data_audit/                     # Generated visualizations
│
├── evaluation/                         # Training and model code
│   ├── model_arc.py                    # ArcFace models (ResNet + ArcFace Head)
│   ├── train_arc.py                    # ArcFace training script
│   └── inference_arc.py                # ArcFace inference utilities
│
├── workspace/some_exp/genus/hald_assignment/StudyCase/
│   ├── _annotations.coco.json          # Original COCO annotations
│   └── images/                         # Original 1920×1080 images
│
├── yolo_superclass_dataset/            # Generated YOLO dataset
│   ├── images/train/                   # Training images
│   ├── images/val/                     # Validation images
│   ├── labels/train/                   # YOLO format labels
│   ├── labels/val/
│   └── dataset.yaml                    # YOLO config
│
├── arcface_dataset/                    # Generated ArcFace dataset
│   ├── train/                          # class1/, class2/, ... (cropped 128×128)
│   ├── test/                           # Test set (20%)
│   ├── dataset_info.json               # Dataset statistics
│   └── class_mapping.json              # Class name to index mapping
│
├── yolo_cascade_training/              # YOLO training outputs
│   └── yolo_superclass_plankton/
│       └── weights/best.pt             # Best YOLO model
│
├── arcface_models/                     # ArcFace training outputs
│   ├── best_model.pth                  # Best ArcFace model (F1-Macro)
│   ├── class_prototypes.pth            # Class prototypes for inference
│   ├── classification_report.txt       # Per-class metrics
│   ├── confusion_matrix.png            # Confusion heatmap
│   ├── training_history.png            # Training curves
│   └── embedding_visualization_umap.png # Embedding space
│
├── create_superclass_yolo_dataset.py   # COCO → YOLO super-class
├── prepare_arcface_dataset.py          # COCO → Cropped ArcFace dataset
├── generate_prototypes.py              # Generate class prototypes
├── cascade_inference.py                # Unified inference pipeline
├── train_yolo_cascade.sh               # YOLO training script
└── split_dataset.py                    # Dataset splitting utilities
```

## 🔬 Technical Details

### Stage 1: YOLO Detection

**Why Single Super-Class?**
- Simplifies detection (detect ANY plankton object)
- Optimizes for **high recall** (don't miss any objects)
- False positives acceptable (Stage 2 filters them out)

**Key Parameters**:
- Low confidence threshold (0.15) for high recall
- Full 1920×1080 resolution (no downsampling)
- YOLOv10-Large for better small object detection

### Stage 2: ArcFace Identification

**Why Sub-Center ArcFace?**
- Handles **intra-class variance** (K=5 sub-centers per class)
- Standard ArcFace would struggle with morphological variations
- Allows multiple "modes" per species

**Architecture**:
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Embedding**: 512-dimensional L2-normalized
- **Loss**: Sub-Center ArcFace (K=5, s=30, m=0.35)
- **Optimizer**: AdamW (better than SGD for metric learning)
- **Scheduler**: CosineAnnealingLR (smooth decay)

**Training Strategy**:
- Hard labels (no label smoothing) - maintains angular margins
- Aggressive augmentation (rotation, affine, color, erasing)
- Lower angular margin (m=0.35) for similar species
- Lower scale (s=30) for stable gradients

**Inference**:
- Discard ArcFace Head (no longer needed)
- Extract L2-normalized embeddings
- Match against class prototypes using cosine similarity
- Confidence threshold (0.6) for final predictions

## 📈 Evaluation Metrics

### Why Multiple Metrics?

**Accuracy alone is misleading** for imbalanced datasets:
- A model could get 90% accuracy by predicting only common classes
- Rare species would be ignored

### Metrics Used

1. **Top-1 Accuracy**: % of exactly correct predictions
   - Good for overall performance
   - Can be misleading with class imbalance

2. **Top-5 Accuracy**: % where correct class is in top 5
   - Shows if model is "close" when wrong
   - High Top-5 (>95%) means model understands features

3. **F1-Macro**: Average F1 across all classes (unweighted)
   - **Best metric for imbalanced data**
   - Treats rare and common classes equally
   - Used for model selection

4. **F1-Weighted**: F1 weighted by class frequency
   - Shows performance on common classes
   - Usually higher than F1-Macro

5. **Precision**: % of predictions that are correct
   - Important for production (avoid misidentification)

6. **Recall**: % of instances detected
   - Important for rare species (don't miss them)

### Interpreting Results

**Example from this project**:
```
Accuracy:     94.83%   ✅ Excellent overall
Top-5 Acc:    98.28%   ✅ Model is "close" when wrong
F1-Macro:     0.8391   ⚠️ Rare classes struggling
F1-Weighted:  0.9484   ✅ Common classes excellent
```

**Interpretation**:
- Overall performance is excellent
- Common species identified nearly perfectly
- Rare species (1-10 samples) have ~60-80% accuracy
- Gap between F1-Macro and F1-Weighted indicates imbalance

## 🎓 Key Improvements from Baseline

### Original Configuration
- SGD optimizer
- ResNet18 backbone
- Limited augmentation
- Accuracy-based model selection
- **Result**: 78% accuracy

### Improved Configuration
- ✅ AdamW optimizer (better convergence)
- ✅ ResNet50 backbone (25M vs 11M parameters)
- ✅ Strong augmentation (rotation, affine, erasing)
- ✅ F1-Macro model selection (handles imbalance)
- ✅ Optimized ArcFace parameters (K=5, s=30, m=0.35)
- ✅ CosineAnnealingLR (smooth decay)
- **Result**: 94.83% accuracy, 0.8391 F1-Macro

**Improvement**: +16.83% accuracy, significantly better rare class performance

## 🔧 Hyperparameter Tuning Guide

### If Accuracy is Low (<85%)
- Increase augmentation strength
- Train longer (100+ epochs)
- Use larger backbone (ResNet50 → ResNet101)
- Lower learning rate (0.0001 → 0.00005)

### If Rare Classes Struggle (F1-Macro << Accuracy)
- Increase sub-centers (K=5 → K=10)
- Augment rare classes more aggressively
- Use class-weighted loss
- Reduce angular margin (m=0.35 → m=0.25)

### If Similar Species Confused (Low Precision)
- Increase angular margin (m=0.35 → m=0.5)
- Add more training data for confused pairs
- Use hard negative mining
- Increase embedding dimension (512 → 1024)

### If Training is Unstable
- Lower learning rate
- Use gradient clipping
- Increase batch size
- Lower scale parameter (s=30 → s=20)

## 📚 References

### Papers
- **ArcFace**: [Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **Sub-Center ArcFace**: [Boosting Face Recognition by Large-scale Noisy Web Faces](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)
- **YOLOv10**: [Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)

### Tools
- [Ultralytics YOLOv10](https://github.com/THU-MIG/yolov10)
- [PyTorch](https://pytorch.org/)
- [Albumentations](https://albumentations.ai/)

## 👨‍💻 Author

**Rohit Mishra**
- Email: rohit225151@rediffmail.com
- GitHub: [@rohitmishra94](https://github.com/rohitmishra94)

## 📄 License

MIT License

## 🙏 Acknowledgments

- Claude Code for development assistance
- Hadl assignment case study for problem definition
- ArcFace and YOLOv10 authors for excellent papers

---

**Generated with Claude Code** 🤖
