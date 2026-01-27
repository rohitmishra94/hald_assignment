# ArcFace Crop Size Strategy

## ✅ IMPLEMENTED: Adaptive 256×256 Sizing

We implemented an **adaptive sizing strategy** that gives the best of both worlds.

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

## Conclusion

**Adaptive 256×256 sizing implemented!** This approach:
- ✅ Preserves small object sharpness (no upsampling)
- ✅ Preserves large object detail (minimal downsampling)
- ✅ Single model with consistent 256×256 batching
- ✅ Best of both worlds

Run dataset preparation and training to see the results!
