# Improvement Suggestions for Plankton Detection System

## 1. Feature Pyramid Network (FPN) for Small Object Detection

### Problem Analysis
- **43% of objects are <20×20 pixels** (2,175 out of 5,050 annotations)
- Mostly Chlorella sp: smallest objects are 9×10 pixels
- At standard YOLO P3 level (stride 8), these objects are only 1-2 feature cells
- Current YOLO recall: 85.9% (could be improved)

### Proposed Solution: Add P2 Feature Level

#### Technical Details
- **Current YOLO Feature Maps**:
  - P3: 1920/8 = 240×135 (smallest standard level)
  - P4: 1920/16 = 120×67
  - P5: 1920/32 = 60×34

- **Proposed P2 Addition**:
  - P2: 1920/4 = 480×270 feature map
  - 10×10 pixel object → 2.5×2.5 feature cells (vs 1.25×1.25 at P3)
  - Better spatial resolution for tiny Chlorella sp

#### Implementation Steps

1. **Modify YOLO Architecture**:
```python
# Add P2 detection head
model.model[-1].anchors = [
    [10,13, 16,30, 33,23],      # P2 - tiny objects (NEW)
    [30,61, 62,45, 59,119],     # P3 - small objects
    [116,90, 156,198, 373,326]  # P4 - medium/large
]
```

2. **Adjust Training Configuration**:
```yaml
# Multi-scale training for better small object learning
scales: [1280, 1920, 2560]
mosaic: 1.0
copy_paste: 0.3  # Augment small objects
```

3. **Optimize Anchor Boxes**:
```bash
# Auto-anchor for dataset
yolo detect train --data dataset.yaml --auto-anchor
```

#### Expected Benefits
- Improved recall: 85.9% → >95%
- Better localization for Chlorella sp
- Reduced false negatives for small objects
- Maintain high precision

---

## 2. Alternative: Patch-Based Detection (SAHI-style)

### Concept
Instead of modifying architecture, process image in overlapping patches:

1. **Divide 1920×1080 into 640×640 patches** with 20% overlap
2. Run YOLO on each patch at full resolution
3. Merge detections with NMS
4. Benefits:
   - No architecture changes needed
   - Preserves full resolution for small objects
   - Proven effective in satellite imagery

### Implementation
```python
def patch_based_inference(image, model, patch_size=640, overlap=0.2):
    patches = create_overlapping_patches(image, patch_size, overlap)
    all_detections = []

    for patch in patches:
        detections = model(patch)
        all_detections.extend(detections)

    final_detections = nms(all_detections, iou_threshold=0.5)
    return final_detections
```

---

## 3. Hybrid Approach: Cascade with Attention

### Stage 1.5: Region Proposal Network
Add intermediate stage between YOLO and ArcFace:

1. YOLO detects general regions
2. **NEW: Attention module focuses on dense regions**
3. Re-run detection at 2x resolution on dense areas
4. Pass to ArcFace for classification

Benefits:
- Computational efficiency (only upsample where needed)
- Better handling of clustered small objects
- Adaptive to image content

---

## 4. Data-Centric Improvements

### Augmentation Strategy for Small Objects
```python
# Targeted augmentation for Chlorella sp
if class_name == 'Chlorella sp' and bbox_area < 400:
    # Apply stronger augmentation
    augmented = A.Compose([
        A.RandomScale(scale_limit=(0.5, 2.0)),  # Make some bigger
        A.RandomBrightnessContrast(p=0.8),
        A.GaussNoise(p=0.5),
    ])(image=crop, bboxes=[bbox])
```

### Synthetic Data Generation
- Generate synthetic Chlorella sp at various scales
- Use copy-paste augmentation more aggressively
- Create "difficult" training samples

---

## 5. Training Strategy Optimizations

### Loss Function Modifications
```python
# Weighted loss for small objects
def weighted_loss(predictions, targets):
    weights = torch.where(
        targets['area'] < 400,
        2.0,  # Double weight for small objects
        1.0
    )
    return base_loss * weights
```

### Hard Negative Mining
- Focus training on difficult Chlorella sp samples
- Use OHEM (Online Hard Example Mining)
- Gradient accumulation for small objects

---

## 6. Post-Processing Improvements

### Size-Aware NMS
```python
def size_aware_nms(detections, size_threshold=400):
    small_objects = [d for d in detections if d.area < size_threshold]
    large_objects = [d for d in detections if d.area >= size_threshold]

    # Different NMS thresholds
    small_nms = nms(small_objects, iou_threshold=0.3)  # Lower threshold
    large_nms = nms(large_objects, iou_threshold=0.5)

    return small_nms + large_nms
```

### Confidence Calibration
- Lower confidence threshold for known small classes
- Class-specific thresholds based on object size distribution

---

## Implementation Priority

1. **High Priority** (Quick wins):
   - Size-aware NMS
   - Class-specific confidence thresholds
   - Targeted augmentation for Chlorella sp

2. **Medium Priority** (Moderate effort):
   - Patch-based detection
   - Weighted loss function
   - Auto-anchor optimization

3. **Low Priority** (Major changes):
   - P2 feature pyramid
   - Hybrid cascade with attention
   - Architecture modifications

---

## Performance Targets

Current:
- YOLO mAP: 91.6%
- YOLO Recall: 85.9%
- ArcFace Accuracy: 98.21%

After improvements:
- YOLO mAP: >93%
- YOLO Recall: >95% (especially for Chlorella sp)
- End-to-end accuracy: >99% for objects >20×20
- End-to-end accuracy: >90% for objects <20×20

---

## 7. Adaptive Crop Sizes for ArcFace Based on BBox Area

### Current Problem
- **Fixed 128×128 crops for all objects**:
  - Small objects (10×10): Upsampled 12.8× → blurry, loss of detail
  - Large objects (200×200): Downsampled 0.64× → loss of fine features
- We have bbox area information but don't use it!

### Proposed Solution: Area-Based Adaptive Cropping

#### Crop Size Strategy
```python
def get_adaptive_crop_size(bbox_area):
    """Select crop size based on object area"""
    if bbox_area < 400:      # <20×20 (mainly Chlorella sp)
        return 64, 64         # 6.4× upsampling vs 12.8× currently
    elif bbox_area < 2500:    # 20×20 to 50×50
        return 128, 128       # Current standard
    elif bbox_area < 10000:   # 50×50 to 100×100
        return 192, 192       # Preserve more detail
    else:                     # >100×100 (Euplotes, Tintinnopsis)
        return 256, 256       # No downsampling, full detail
```

#### Implementation Approach
```python
def adaptive_crop_and_pad(image, bbox, max_size=256):
    """Crop with adaptive sizing and aspect ratio preservation"""
    area = bbox['width'] * bbox['height']
    target_size = get_adaptive_crop_size(area)

    # Extract with padding
    cropped = extract_with_context(image, bbox, padding=0.2)

    # Resize maintaining aspect ratio
    h, w = cropped.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)

    # Don't upscale more than 8x (quality limit)
    scale = min(scale, 8.0)

    resized = cv2.resize(cropped, None, fx=scale, fy=scale)

    # Center pad to target size for batching
    padded = center_pad_to_square(resized, max_size)

    return padded
```

### Benefits by Species Category

| Species Type | Current (128×128) | Adaptive | Improvement |
|-------------|------------------|----------|-------------|
| Chlorella sp (10×10) | 12.8× upsampling | 6.4× upsampling (64×64) | 50% less interpolation |
| Small species (30×30) | 4.3× upsampling | 4.3× (unchanged) | - |
| Medium species (80×80) | 1.6× upsampling | 2.4× to 192×192 | More detail preserved |
| Large species (200×200) | 0.64× downsampling | 1.28× to 256×256 | No detail loss |

### Multi-Scale Training Strategy

#### Option 1: Size-Specific Models
```python
models = {
    'tiny': ArcFaceModel(input_size=64),
    'small': ArcFaceModel(input_size=128),
    'medium': ArcFaceModel(input_size=192),
    'large': ArcFaceModel(input_size=256)
}

# Route based on bbox area during inference
def inference(crop, bbox_area):
    if bbox_area < 400:
        return models['tiny'](crop)
    # ... etc
```

#### Option 2: Single Model with Padding (Recommended)
- Train single model with max size (256×256)
- Smaller objects centered with padding
- Maintains single model simplicity
- Add position encoding for size awareness

### Expected Improvements

1. **Chlorella sp (43% of dataset)**:
   - Current: 78% accuracy (blurry from 12.8× upsampling)
   - Expected: 85-90% (sharper with 6.4× upsampling)

2. **Large species (Euplotes, Tintinnopsis)**:
   - Current: 95% accuracy (some detail loss)
   - Expected: 98%+ (full detail preserved)

3. **Overall Impact**:
   - F1-Macro: 0.8995 → 0.92+
   - Especially helps extreme size classes

### Implementation Priority
**Medium-High Priority**: Relatively easy to implement with significant expected gains

### Code Changes Required
1. Modify `prepare_arcface_dataset.py` to use adaptive sizes
2. Update `train_arc.py` to handle variable input sizes
3. Adjust `generate_prototypes.py` for multi-scale embeddings

## 8. YOLO11 for Improved Detection Performance

### What is YOLO11?
Released September 2024, YOLO11 is the latest YOLO architecture from Ultralytics with significant improvements:

#### Key Features
- **Improved Architecture**: Enhanced backbone and feature extraction
- **Better Small Object Detection**: Improved feature pyramid for tiny objects
- **Training Efficiency**: Faster convergence with better optimization
- **Edge Optimization**: Designed for efficient deployment on edge devices
- **Latest Research**: Incorporates 2024 computer vision advances

#### Requirements
```bash
pip install -U ultralytics>=8.3.0  # YOLO11 support
```

### Comparison with YOLOv10

| Feature | YOLOv10-Large | YOLO11-Large |
|---------|---------------|--------------|
| Release Date | May 2024 | September 2024 |
| Architecture | Dual-head | Enhanced |
| Training Speed | Baseline | Faster |
| Small Object Detection | Good | Better |
| Pretrained Weights | Available | Available |

### Implementation in Project

```bash
# Train YOLO11 for single super-class detection (Stage 1)
bash train_yolo26_cascade.sh

# Configuration:
- Model: yolo11l.pt
- Classes: 1 (super-class "plankton")
- Resolution: 1920×1080
- Purpose: High recall detection for ArcFace Stage 2
```

### Expected Benefits

1. **Better Training**: Faster convergence with improved optimization
2. **Better Small Object Recall**: Improved detection for Chlorella sp (<20×20)
3. **Reduced False Negatives**: Better for production counting
4. **Edge Deployment**: More efficient for embedded systems
5. **Drop-in Replacement**: Compatible with existing cascade pipeline

### Comparison with YOLOv10

| Metric | YOLOv10-Large | YOLO11-Large |
|--------|---------------|--------------|
| **mAP@0.5** | 91.6% | TBD (training) |
| **Recall** | 85.9% | TBD (training) |
| **Architecture** | May 2024 | Sep 2024 (latest) |
| **Small Objects** | Good | Better |
| **Training Speed** | Baseline | Faster |

Test with identical settings:
- Same dataset (yolo_superclass_dataset)
- Same resolution (1920×1080)
- Same evaluation metrics

### Why Cascade Remains Superior

**Cascade (YOLO + ArcFace) Advantages:**
- ✅ **Proven 98.21% accuracy** on 39-class fine-grained classification
- ✅ Handles intra-class variance with Sub-Center ArcFace
- ✅ Can update classification without retraining detection
- ✅ Modular: Each stage optimized for its specific task
- ✅ Better for fine-grained species with high similarity

**Key Insight**: Fine-grained classification (39 similar species) is fundamentally different from detection (1 class). The cascade approach leverages this by using:
- YOLO: Optimized for high recall detection (91.6% mAP)
- ArcFace: Optimized for metric learning (98.21% accuracy)

### Recommendation

**Use YOLO11 for Cascade Stage 1**:
- Drop-in replacement for YOLOv10
- Better small object detection expected
- Compatible with existing ArcFace Stage 2
- Continue with proven 98.21% accuracy cascade pipeline

### Implementation File
- **`train_yolo26_cascade.sh`** - YOLO11 Stage 1 training script

---

## Notes
- Current training at 1920 resolution is good - keep it
- Consider TTA (Test Time Augmentation) for critical applications
- Monitor GPU memory with P2 addition (may need batch size reduction)
- Adaptive cropping could be the easiest high-impact improvement
- YOLO11 improvements are particularly beneficial for small objects in dense plankton images