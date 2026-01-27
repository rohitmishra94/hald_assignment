# ArcFace Crop Size Strategy

## ❌ EXPERIMENT FAILED: Adaptive 256×256 Sizing

We tested an **adaptive sizing strategy** but it performed significantly worse than the simple 128×128 approach.

## ✅ FINAL DECISION: Use Fixed 128×128

Reverting to the proven fixed 128×128 configuration.

### Dataset Characteristics
- **43% of objects are <20×20 pixels** (2,175 out of 5,050 annotations)
- Smallest objects: 9×10 pixels (Chlorella sp)
- Largest objects: ~200×200 pixels (Euplotes, Tintinnopsis)

## How It Works

### Adaptive Sizing Strategy (IMPLEMENTED)

```python
def extract_and_resize_object(image, bbox, target_size=(256, 256), padding=0.1):
    # 1. Extract bbox with padding
    cropped = extract_bbox_with_padding(image, bbox, padding)

    # 2. Adaptive resize
    if cropped.size > target_size:
        # Large object: resize down to 256×256
        result = cv2.resize(cropped, target_size, cv2.INTER_AREA)
    else:
        # Small object: keep original size, pad to 256×256
        result = center_pad(cropped, target_size)

    return result
```

### Results by Object Size

| Object Size | Old (128×128 fixed) | New (256×256 adaptive) | Improvement |
|------------|---------------------|------------------------|-------------|
| 10×10 px   | 12.8× upsampling (blurry) | **No upsampling** (sharp!) | ✅ Much better |
| 50×50 px   | 2.6× upsampling | **No upsampling** | ✅ Better |
| 100×100 px | 1.3× upsampling | **No upsampling** | ✅ Better |
| 200×200 px | 0.64× downsampling (loss) | 1.28× upsampling | ✅ Better |
| 300×300 px | 0.43× downsampling (loss) | 0.85× downsampling | ✅ Better |

## Benefits of Adaptive Sizing

### ✅ Advantages Over Fixed 128×128

1. **Small objects stay sharp** (43% of dataset)
   - No aggressive upsampling blur
   - Chlorella sp (10×10 px) keeps all original detail
   - Better feature preservation

2. **Large objects get full resolution**
   - Euplotes, Tintinnopsis preserve fine details
   - Less downsampling loss
   - Better discrimination between similar species

3. **Optimal information density**
   - Every object uses appropriate resolution
   - No wasted pixels on excessive upsampling
   - No lost detail from excessive downsampling

4. **Single model simplicity**
   - All images are 256×256 for batching
   - No need for multiple models
   - Position-aware via padding

### ⚠️ Trade-offs

**Increased compute:**
- 4× more pixels than 128×128 (256²/128² = 4)
- Slower training (may need to reduce batch size)
- More GPU memory required

**Expected outcome:**
- Better accuracy on extreme sizes (very small and very large)
- Slightly longer training time
- May need batch_size=16 instead of 32

## Implementation Details

### Files Modified

1. **`prepare_arcface_dataset.py`**:
   - Changed `target_size` from (128,128) to (256,256)
   - Updated `extract_and_resize_object()` with adaptive logic
   - Small objects: padded to center
   - Large objects: resized down

2. **`evaluation/train_arc.py`**:
   - Removed `transforms.Resize()` (images already correct size)
   - Kept all augmentation transforms
   - Works with 256×256 inputs automatically

### Next Steps

1. ✅ **Fixed validation set issue** (single-sample classes duplicated)
2. ✅ **Implemented adaptive 256×256 sizing**
3. 🔄 **Re-run dataset preparation**: `python prepare_arcface_dataset.py`
4. 🔄 **Train with adaptive 256×256**: `cd evaluation && python train_arc.py`
5. ⏳ **Compare with baseline 98.21%** (128×128 fixed)
6. ⏳ **Analyze per-class improvements**

### Expected Improvements

Classes most likely to benefit:
- **Chlorella sp** (10×10 px, 43% of dataset): sharper images
- **Euplotes sp** (large, detailed): full resolution preserved
- **Tintinnopsis sp** (large, detailed): better fine features
- **Medium species** (50-100 px): better detail preservation

Target metrics:
- F1-Macro: 0.8995 → **0.92+** (especially rare small classes)
- Chlorella sp accuracy: improve on current performance
- Large species: near-perfect accuracy

## Experimental Results: Adaptive 256×256 Failed

### Performance Comparison

| Metric | 128×128 Fixed | 256×256 Adaptive | Change |
|--------|--------------|------------------|---------|
| **Accuracy** | 98.21% | 92.63% | **-5.58%** ❌ |
| **Top-5 Accuracy** | 99.31% | 96.49% | **-2.82%** ❌ |
| **F1-Macro** | 0.8995 | 0.7196 | **-0.1799** ❌ |
| **F1-Weighted** | 0.9813 | 0.9233 | **-0.0580** ❌ |
| **Precision** | 0.8941 | 0.7049 | **-0.1892** ❌ |
| **Recall** | 0.9079 | 0.7441 | **-0.1638** ❌ |

### Classes That Failed with 256×256

**10 classes completely failed** (0% F1-score):
- Anisonema sp (10 samples) - was working with 128×128
- Cyclidium sp (1 sample)
- Gyrodinium sp (1 sample)
- Oxyrrhis sp (5 samples) - was working with 128×128
- Skeletonema sp (4 samples) - was working with 128×128
- Spirulina sp (1 sample)
- Plus severe degradation: Chlamydomonas sp (18% F1), Chaetoceros sp (19% F1)

Previously with 128×128: Only 3 single-sample classes failed.

### Why Adaptive 256×256 Failed

1. **"Needle in Haystack" Problem**
   - Small objects (10×10 px) became tiny specks in 256×256 black canvas
   - Too much padding (up to ~120px on each side)
   - Model learned to focus on padding patterns instead of object features

2. **Position Overfitting**
   - All small objects centered in same position
   - Model overfitted to object location rather than features
   - Lost spatial context information

3. **Augmentation Issues**
   - RandomErasing might delete entire small objects
   - Translation/rotation pushed small objects to canvas edges
   - Augmentation designed for 128×128 didn't scale well

4. **Training Dynamics**
   - 4× more pixels but same model capacity
   - Batch normalization affected by 4× pixel increase
   - Learning rate tuned for 128×128 might be inappropriate

5. **Information Density Mismatch**
   - Small objects: mostly black padding (low information)
   - Large objects: rich details (high information)
   - Model struggled to handle this variance

## Conclusion: Stick with 128×128

**Fixed 128×128 is the winner:**
- ✅ **98.21% accuracy** (proven)
- ✅ Balanced upsampling/downsampling for all sizes
- ✅ Simpler, more stable training
- ✅ Better for 43% of dataset (small objects)
- ✅ Computationally efficient

**Key Lesson Learned:**
Theoretical improvements (no upsampling) don't always translate to practical gains. The complexity introduced by adaptive sizing and excessive padding outweighed the benefits of preserving small object resolution.

## Next Steps

1. ✅ **Reverted to 128×128 fixed sizing**
2. 🔄 **Re-run dataset preparation**: `python prepare_arcface_dataset.py`
3. 🔄 **Re-train model**: `cd evaluation && python train_arc.py`
4. ⏳ **Expect ~98% accuracy return**

The 128×128 configuration with simple resize is the optimal choice for this dataset.
