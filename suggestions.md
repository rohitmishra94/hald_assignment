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

## Notes
- Current training at 1920 resolution is good - keep it
- Consider TTA (Test Time Augmentation) for critical applications
- Monitor GPU memory with P2 addition (may need batch size reduction)