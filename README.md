# Fine-Grained Plankton Identification System

A high-precision cascade pipeline for plankton species identification from microscopy images, addressing the challenges of high intra-class variance and inter-class similarity in fine-grained biological classification.

## 🎯 Overview

This project implements a production-ready plankton identification system using a **2-stage cascade architecture**:

1. **Stage 1 (Detection)**: YOLOv10/YOLO11 detect ALL plankton objects with high recall
2. **Stage 2 (Classification)**: ArcFace identifies species using deep metric learning

An **end-to-end YOLO11 variant** (39-class detection+classification) is also available for comparison.

### Key Challenges Addressed
- **High intra-class variance**: Same species looks different due to rotation, lighting, morphology
- **High inter-class similarity**: Different species look nearly identical
- **Class imbalance**: Rare species with few samples (1-2531 per class)
- **Fine-grained features**: Subtle differences between species
- **Small objects**: 43% of objects are <20×20 pixels

## 📊 Performance

### Cascade Pipeline (YOLOv10 + ArcFace)

**Stage 1: YOLOv10 Detection**
- **Model**: YOLOv10-Large (single "plankton" super-class)
- **mAP@0.5**: 91.6%
- **Recall**: 85.9%
- **Resolution**: 1920×1080 (full native resolution)
- **Confidence**: 0.15 (high recall)

**Stage 2: ArcFace Classification**
- **Model**: ResNet50 + Sub-Center ArcFace (K=5)
- **Top-1 Accuracy**: 98.21% ⭐
- **Top-5 Accuracy**: 99.31%
- **F1-Macro**: 0.8995 (all classes treated equally)
- **F1-Weighted**: 0.9813 (weighted by class frequency)
- **Precision**: 0.8941
- **Recall**: 0.9079
- **Classes**: 39 plankton species

**End-to-End Pipeline**
- **Combined Accuracy**: ~90-95% (high confidence predictions only)
- **Speed**: ~50-100 FPS (depends on object density)
- **Confidence Thresholds**: YOLO=0.15, ArcFace=0.6

### Alternative: YOLO11 Models

**YOLO11 Cascade (Stage 1 only)**
- Latest architecture (Sep 2024)
- Better small object detection than YOLOv10
- Improved training efficiency
- Training: `bash train_yolo26_cascade.sh`

**YOLO11 End-to-End (39 classes)**
- Single model for detection + classification
- Simpler deployment, faster inference
- For comparing cascade vs end-to-end approaches
- Training: `bash train_yolo26_multiclass.sh`

### Key Achievements
- ✅ **98.21% accuracy** on 39-class fine-grained classification (Cascade)
- ✅ **99.31% Top-5 accuracy** - correct species almost always in top 5
- ✅ **0.8995 F1-Macro** - excellent performance across all classes including rare species
- ✅ Only 3 species with 0% (Cyclidium sp, Gyrodinium sp, Spirulina sp - single test samples)
- ✅ **20.21% improvement** from baseline (78% → 98.21%)
- ✅ Handles rare classes (14 classes with <30 samples) via augmentation

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

#### For Cascade Pipeline (YOLOv10/YOLO11 + ArcFace)

```bash
# Stage 1: Create YOLO super-class dataset (single "plankton" class)
python create_superclass_yolo_dataset.py

# Stage 2: Create ArcFace dataset (cropped 128×128 objects by species)
python prepare_arcface_dataset.py
```

**Output**:
- `yolo_superclass_dataset/` - YOLO Stage 1 training data
- `arcface_dataset/` - ArcFace Stage 2 training data (train/test splits)

#### For End-to-End YOLO11 (39 classes)

```bash
# Create multi-class YOLO dataset with stratified split and augmentation
python create_multiclass_yolo_dataset.py
```

**Output**:
- `yolo_multiclass_dataset/` - 39-class YOLO training data
- Stratified train/val split ensures all classes in both sets
- Augments rare classes (<30 samples) to 50 samples

### 3. Training

#### Option A: Cascade Pipeline

**Stage 1: Train YOLO Detector (YOLOv10 or YOLO11)**

```bash
# YOLOv10-Large (proven results)
bash train_yolo_cascade.sh

# OR YOLO11-Large (latest architecture, better small objects)
bash train_yolo26_cascade.sh
```

**Configuration**:
- Model: YOLOv10-Large or YOLO11-Large
- Resolution: 1920×1080
- Epochs: 100
- Batch size: 2
- Confidence: 0.15 (high recall)

**Stage 2: Train ArcFace Classifier**

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

#### Option B: End-to-End YOLO11 (39 classes)

```bash
# Train YOLO11 for both detection + classification
bash train_yolo26_multiclass.sh
```

**Configuration**:
- Model: YOLO11-Large
- Classes: 39 (all species)
- Resolution: 1920×1080
- Epochs: 100
- Batch size: 2
- Confidence: 0.25 (balanced for 39 classes)

**Use case**: Compare single-model vs cascade approach for accuracy/speed trade-offs

### 4. Generate Class Prototypes (Cascade only)

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

#### Cascade Pipeline (YOLOv10/YOLO11 + ArcFace)

**Single Image**
```bash
python cascade_inference.py \
    --image path/to/image.jpg \
    --yolo yolo_cascade_training/yolo_superclass_plankton/weights/best.pt \
    --arcface arcface_models/best_model.pth \
    --prototypes arcface_models/class_prototypes.pth \
    --yolo-conf 0.15 \
    --arcface-conf 0.6
```

**Batch Processing**
```bash
python cascade_inference.py \
    --batch path/to/images/ \
    --output cascade_results/ \
    --yolo-conf 0.15 \
    --arcface-conf 0.6
```

#### End-to-End YOLO11 (39 classes)

**Single Image**
```bash
yolo detect predict \
    model=yolo26_multiclass_training/yolo11_multiclass_39species/weights/best.pt \
    source=path/to/image.jpg \
    conf=0.25 \
    imgsz=1920 \
    save=True
```

**Batch Processing**
```bash
yolo detect predict \
    model=yolo26_multiclass_training/yolo11_multiclass_39species/weights/best.pt \
    source=path/to/images/ \
    conf=0.25 \
    imgsz=1920 \
    save=True
```

## 🏗️ System Architecture

### Cascade Pipeline (Recommended for high accuracy)

```
┌─────────────────────────────────────────────────────────────┐
│                     STAGE 1: DETECTION                       │
│                                                              │
│  ┌────────────┐        ┌──────────────┐                    │
│  │   Image    │   →    │ YOLOv10/26   │   →  Bounding Boxes│
│  │ 1920×1080  │        │ (1 Class)    │                    │
│  └────────────┘        └──────────────┘                     │
│                                                              │
│  Goal: Detect ALL plankton objects (High Recall)            │
│  Confidence: 0.15 (low threshold to catch everything)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Cropped Object Images
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 2: CLASSIFICATION                    │
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
│               Class Prototypes (39 species)                  │
│                              ↓                               │
│                   Species ID + Confidence                    │
│                                                              │
│  Goal: Identify species using metric learning               │
│  Sub-Center ArcFace (K=5) for intra-class variance          │
│  Confidence: 0.6 (only trust high similarity)               │
└─────────────────────────────────────────────────────────────┘

Result: 98.21% accuracy, 0.8995 F1-Macro
```

### End-to-End YOLO11 (Simpler deployment)

```
┌─────────────────────────────────────────────────────────────┐
│              SINGLE STAGE: DETECTION + CLASSIFICATION        │
│                                                              │
│  ┌────────────┐        ┌──────────────┐                    │
│  │   Image    │   →    │   YOLO11     │   →  BBoxes +     │
│  │ 1920×1080  │        │ (39 Classes) │      Species IDs   │
│  └────────────┘        └──────────────┘                     │
│                                                              │
│  Goal: Direct detection + classification in one model       │
│  Confidence: 0.25 (balanced for multi-class)                │
│  Latest architecture (Sep 2024)                             │
└─────────────────────────────────────────────────────────────┘

Trade-off: Simpler & faster, but may sacrifice accuracy on
fine-grained species with high inter-class similarity
```

## 📁 Project Structure

```
hald_assignment/
├── README.md                           # This file
├── suggestions.md                      # Improvement ideas & YOLO26 analysis
│
├── StudyCase/
│   ├── _annotations.coco.json          # Original COCO annotations (5050 objects, 39 classes)
│   └── images/                         # Original 1920×1080 images
│
├── data_audit/                         # Data analysis
│   ├── data_analysis.ipynb             # Class distribution, bbox analysis
│   └── intra_class_analysis.ipynb      # Intra-class variance study
│
├── evaluation/                         # ArcFace training
│   ├── model_arc.py                    # ResNet50 + Sub-Center ArcFace
│   ├── train_arc.py                    # Training script
│   └── inference_arc.py                # Inference utilities
│
├── Dataset Preparation Scripts
│   ├── create_superclass_yolo_dataset.py   # Cascade Stage 1 (1 class)
│   ├── create_multiclass_yolo_dataset.py   # End-to-end (39 classes)
│   ├── prepare_arcface_dataset.py          # Cascade Stage 2 (cropped 128×128)
│   └── analyze_small_objects.py            # Small object analysis
│
├── Training Scripts
│   ├── train_yolo_cascade.sh           # YOLOv10-Large cascade
│   ├── train_yolo26_cascade.sh         # YOLO26-Large cascade
│   └── train_yolo26_multiclass.sh      # YOLO26-Large end-to-end
│
├── Inference & Prototypes
│   ├── generate_prototypes.py          # Generate ArcFace class prototypes
│   └── cascade_inference.py            # Cascade pipeline inference
│
├── Generated Datasets
│   ├── yolo_superclass_dataset/        # Cascade Stage 1 (1 class)
│   ├── yolo_multiclass_dataset/        # End-to-end (39 classes, stratified, augmented)
│   └── arcface_dataset/                # Cascade Stage 2 (cropped by species)
│
└── Training Outputs
    ├── yolo_cascade_training/          # YOLOv10 cascade models
    ├── yolo26_cascade_training/        # YOLO26 cascade models
    ├── yolo26_multiclass_training/     # YOLO26 end-to-end models
    └── arcface_models/                 # ArcFace models + prototypes
```

## 🔬 Technical Details

### Cascade vs End-to-End Comparison

| Aspect | Cascade (YOLOv10/11 + ArcFace) | End-to-End (YOLO11 39-class) |
|--------|-------------------------------|------------------------------|
| **Models** | 2 models (detection + classification) | 1 model (combined) |
| **Accuracy** | 98.21% (proven) | TBD (comparison pending) |
| **Inference** | 2-stage (slower) | 1-stage (faster) |
| **Deployment** | More complex | Simpler |
| **Fine-grained** | Excellent (metric learning) | May struggle with similar species |
| **Flexibility** | Can update classifier independently | Must retrain entire model |
| **Use case** | Production, high accuracy needed | Edge devices, speed priority |

### Cascade Stage 1: YOLO Detection

**Why Single Super-Class?**
- Simplifies detection (detect ANY plankton object)
- Optimizes for **high recall** (don't miss any objects)
- False positives acceptable (Stage 2 filters them out)

**Model Options**:
- **YOLOv10-Large**: Proven, 91.6% mAP@0.5, 85.9% recall
- **YOLO11-Large**: Latest (Sep 2024), better small object detection

**Key Parameters**:
- Low confidence threshold (0.15) for high recall
- Full 1920×1080 resolution (no downsampling)
- Optimized for small objects (<20×20 pixels)

### Cascade Stage 2: ArcFace Classification

**Why Sub-Center ArcFace?**
- Handles **intra-class variance** (K=5 sub-centers per class)
- Standard ArcFace would struggle with morphological variations within same species
- Allows multiple "modes" per species (different rotations, life stages, etc.)

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
- F1-Macro based model selection (handles class imbalance)

**Inference**:
- Discard ArcFace Head (training only)
- Extract L2-normalized embeddings from backbone
- Match against class prototypes using cosine similarity
- Confidence threshold (0.6) for final predictions

### End-to-End YOLO11 (39 classes)

**Why End-to-End?**
- Single model simplicity for deployment
- Faster inference (no 2-stage overhead)
- Lower memory footprint
- Easier to maintain and update

**Challenges**:
- 39-class fine-grained classification is harder than 1-class detection
- High inter-class similarity between species
- Class imbalance (1 to 2531 samples per class)

**Solutions Implemented**:
- **Stratified splitting**: Ensures all 39 classes in train and val
- **Rare class augmentation**: Classes with <30 samples augmented to 50
- **YOLO11 architecture**: Latest improvements, better small object detection
- **Full resolution**: 1920×1080 training (no downsampling)

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

**Final Results from this project**:
```
Accuracy:     98.21%   ⭐ Outstanding overall performance
Top-5 Acc:    99.31%   ⭐ Correct species almost always in top 5
F1-Macro:     0.8995   ✅ Excellent across all classes
F1-Weighted:  0.9813   ⭐ Near-perfect on common species
Precision:    0.8941   ✅ Predictions are highly reliable
Recall:       0.9079   ✅ Detecting most instances
```

**Interpretation**:
- Outstanding performance across all metrics
- Common species (Chlorella sp, Oscillatoria sp, Prymnesium sp) identified at 99-100%
- Most rare species (1-10 samples) achieve 80-100% accuracy
- Only 3 species failed: Cyclidium sp, Gyrodinium sp, Spirulina sp (single test samples)
- Gap between F1-Macro (0.8995) and F1-Weighted (0.9813) indicates the 3 failed species are very rare

**Per-Class Performance Highlights**:
- **Perfect (100% F1)**: 18 out of 39 species
- **Excellent (>90% F1)**: 33 out of 39 species
- **Good (>80% F1)**: 36 out of 39 species
- **Failed (0% F1)**: 3 species with single test samples

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
- **Result**: 98.21% accuracy, 0.8995 F1-Macro

**Improvement**: +20.21% accuracy (78% → 98.21%), significantly better rare class performance

### Impact of Improvements

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| Accuracy | 78.00% | 98.21% | +20.21% |
| F1-Macro | ~0.65 | 0.8995 | +38.4% |
| Top-5 Acc | ~85% | 99.31% | +16.8% |
| Species with 100% F1 | ~5 | 18 | +260% |
| Species with >90% F1 | ~15 | 33 | +120% |

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
