import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import yaml
import cv2
import numpy as np
from collections import Counter

def augment_image_with_labels(image, labels, img_width, img_height):
    """
    Augment image and update YOLO labels accordingly
    Returns: augmented_image, augmented_labels
    """
    # Random choice of augmentation
    aug_type = np.random.choice(['flip_h', 'flip_v', 'rotate90', 'brightness', 'contrast'])

    aug_image = image.copy()
    aug_labels = labels.copy()

    if aug_type == 'flip_h':
        aug_image = cv2.flip(image, 1)
        # Update x_center: new_x = 1 - old_x
        for i in range(len(aug_labels)):
            aug_labels[i][1] = 1.0 - aug_labels[i][1]

    elif aug_type == 'flip_v':
        aug_image = cv2.flip(image, 0)
        # Update y_center: new_y = 1 - old_y
        for i in range(len(aug_labels)):
            aug_labels[i][2] = 1.0 - aug_labels[i][2]

    elif aug_type == 'rotate90':
        aug_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # Update coordinates for 90° rotation
        for i in range(len(aug_labels)):
            old_x, old_y = aug_labels[i][1], aug_labels[i][2]
            old_w, old_h = aug_labels[i][3], aug_labels[i][4]
            # After 90° CW: (x,y) -> (y, 1-x), (w,h) -> (h,w)
            aug_labels[i][1] = old_y
            aug_labels[i][2] = 1.0 - old_x
            aug_labels[i][3] = old_h
            aug_labels[i][4] = old_w

    elif aug_type == 'brightness':
        factor = np.random.uniform(0.7, 1.3)
        aug_image = np.clip(image * factor, 0, 255).astype(np.uint8)

    elif aug_type == 'contrast':
        factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(image)
        aug_image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

    return aug_image, aug_labels

def create_yolo_multiclass_dataset(
    coco_json_path: str,
    images_dir: str,
    output_dir: str = 'yolo_multiclass_dataset',
    train_split: float = 0.8,
    augment_rare_classes: bool = True,
    min_samples_threshold: int = 30,
    target_samples: int = 50
):
    """
    Create complete YOLO dataset with ALL 39 classes for end-to-end detection + classification

    Creates directory structure:
    yolo_multiclass_dataset/
      ├── images/
      │   ├── train/
      │   └── val/
      ├── labels/
      │   ├── train/
      │   └── val/
      ├── dataset.yaml
      └── class_info.json

    Args:
        coco_json_path: Path to COCO annotations
        images_dir: Path to source images
        output_dir: Output directory
        train_split: Train/val split ratio
        augment_rare_classes: Whether to augment rare classes
        min_samples_threshold: Classes below this get augmented
        target_samples: Target number of samples for rare classes
    """

    print("="*60)
    print("CREATING YOLO MULTI-CLASS DATASET (39 CLASSES)")
    print("="*60)

    # Load COCO annotations
    print(f"\nLoading COCO annotations from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Get categories
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"\nDataset info:")
    print(f"  Classes: {len(categories)}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Images: {len(coco_data['images'])}")

    # Create output directories
    images_train_dir = os.path.join(output_dir, 'images', 'train')
    images_val_dir = os.path.join(output_dir, 'images', 'val')
    labels_train_dir = os.path.join(output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(output_dir, 'labels', 'val')

    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Create image ID to info mapping
    image_map = {img['id']: img for img in coco_data['images']}

    # Group annotations by image
    from collections import defaultdict
    image_to_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_to_annotations[ann['image_id']].append(ann)

    # Split images into train/val with stratification for rare classes
    # Ensure all classes appear in both train and val sets
    import random
    random.seed(42)

    # Group images by their rarest class (to ensure rare classes are stratified)
    from collections import Counter
    image_class_counts = defaultdict(list)

    for img_id in image_to_annotations.keys():
        # Get all classes in this image
        classes_in_image = [categories[ann['category_id']] for ann in image_to_annotations[img_id]]
        # Use the rarest class to guide stratification
        class_counts = Counter([categories[ann['category_id']] for ann in coco_data['annotations']])
        rarest_class = min(classes_in_image, key=lambda c: class_counts[c])
        image_class_counts[rarest_class].append(img_id)

    # Get all image IDs for total count
    all_image_ids = list(image_to_annotations.keys())

    # Stratified split: ensure EVERY class appears in BOTH train and val
    train_image_ids = set()
    val_image_ids = set()

    for class_name, img_ids in image_class_counts.items():
        random.shuffle(img_ids)
        class_count = len(img_ids)

        if class_count == 1:
            # Only 1 sample: put in BOTH train and val (duplicate)
            train_image_ids.update(img_ids)
            val_image_ids.update(img_ids)
        elif class_count == 2:
            # 2 samples: put 1 in train, 1 in val, duplicate the train one to val
            train_image_ids.update([img_ids[0]])
            val_image_ids.update([img_ids[0], img_ids[1]])  # Both in val
        else:
            # 3+ samples: normal stratified split
            n_train_class = max(2, int(class_count * train_split))  # At least 2 in train
            n_val_class = max(1, class_count - n_train_class)       # At least 1 in val

            train_image_ids.update(img_ids[:n_train_class])
            val_image_ids.update(img_ids[n_train_class:n_train_class + n_val_class])

    print(f"\nSplitting dataset:")
    print(f"  Train images: {len(train_image_ids)}")
    print(f"  Val images: {len(val_image_ids)}")

    # Verify all classes appear in both splits
    train_classes = set()
    val_classes = set()
    for img_id in train_image_ids:
        train_classes.update([categories[ann['category_id']] for ann in image_to_annotations[img_id]])
    for img_id in val_image_ids:
        val_classes.update([categories[ann['category_id']] for ann in image_to_annotations[img_id]])

    missing_train = set(categories.values()) - train_classes
    missing_val = set(categories.values()) - val_classes

    if missing_train:
        print(f"\n  ⚠️  WARNING: Classes missing in TRAIN: {missing_train}")
    else:
        print(f"\n  ✅ All {len(categories)} classes present in TRAIN")

    if missing_val:
        print(f"  ⚠️  WARNING: Classes missing in VAL: {missing_val}")
    else:
        print(f"  ✅ All {len(categories)} classes present in VAL")

    print(f"\n  Classes in train: {len(train_classes)}/{len(categories)}")
    print(f"  Classes in val: {len(val_classes)}/{len(categories)}")

    # Create class name to index mapping (0-indexed for YOLO)
    class_names = sorted(categories.values())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    print(f"\nClass mapping (first 10):")
    for i, name in enumerate(class_names[:10]):
        print(f"  {i}: {name}")
    print(f"  ...")

    # Count samples per class in train split for augmentation
    train_class_image_map = defaultdict(list)
    for img_id in train_image_ids:
        for ann in image_to_annotations[img_id]:
            class_name = categories[ann['category_id']]
            train_class_image_map[class_name].append(img_id)

    # Identify rare classes that need augmentation
    rare_classes = {}
    if augment_rare_classes:
        print(f"\n{'='*60}")
        print("AUGMENTATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Classes with <{min_samples_threshold} train samples will be augmented to {target_samples}")
        print(f"\n{'Class':<30} {'Train Samples':<15} {'Action'}")
        print("-"*60)

        for class_name in sorted(class_names):
            train_count = len(set(train_class_image_map[class_name]))
            if train_count < min_samples_threshold:
                needed = max(0, target_samples - train_count)
                rare_classes[class_name] = needed
                print(f"{class_name:<30} {train_count:<15} Augment +{needed}")

        print(f"\nTotal rare classes to augment: {len(rare_classes)}")

    # Process each image
    from tqdm import tqdm

    aug_counter = defaultdict(int)

    for split_name, split_ids, img_dir, lbl_dir in [
        ('train', train_image_ids, images_train_dir, labels_train_dir),
        ('val', val_image_ids, images_val_dir, labels_val_dir)
    ]:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split...")
        print(f"{'='*60}")

        for img_id in tqdm(split_ids, desc=f"{split_name}"):
            img_info = image_map[img_id]

            # Handle file_name that may include 'images/' prefix
            file_name = img_info['file_name']
            if file_name.startswith('images/'):
                file_name = file_name.replace('images/', '', 1)

            src_img_path = os.path.join(images_dir, file_name)

            if not os.path.exists(src_img_path):
                print(f"Warning: Image not found: {src_img_path}")
                continue

            # Copy image (use only the filename, not the path with 'images/')
            dst_img_path = os.path.join(img_dir, os.path.basename(file_name))
            shutil.copy2(src_img_path, dst_img_path)

            # Convert annotations to YOLO format
            img_width = img_info['width']
            img_height = img_info['height']

            annotations = image_to_annotations[img_id]

            # Create YOLO label file
            label_filename = Path(os.path.basename(file_name)).stem + '.txt'
            label_path = os.path.join(lbl_dir, label_filename)

            # Prepare YOLO labels
            yolo_labels = []
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height] in pixels

                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width_norm = bbox[2] / img_width
                height_norm = bbox[3] / img_height

                # Get class index (0-indexed)
                cat_name = categories[ann['category_id']]
                class_idx = class_to_idx[cat_name]

                yolo_labels.append([class_idx, x_center, y_center, width_norm, height_norm])

            # Write original labels
            with open(label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

            # Augmentation for train split only
            if split_name == 'train' and augment_rare_classes:
                # Check if this image contains rare classes
                img_rare_classes = set()
                for ann in annotations:
                    cat_name = categories[ann['category_id']]
                    if cat_name in rare_classes and aug_counter[cat_name] < rare_classes[cat_name]:
                        img_rare_classes.add(cat_name)

                if img_rare_classes:
                    # Load image for augmentation
                    image = cv2.imread(src_img_path)
                    if image is None:
                        continue

                    # Determine how many augmentations to create
                    max_needed = max(rare_classes[c] - aug_counter[c] for c in img_rare_classes)
                    n_augmentations = min(5, max_needed)  # Max 5 augmentations per image

                    for aug_idx in range(n_augmentations):
                        # Augment image with labels
                        aug_image, aug_labels = augment_image_with_labels(
                            image, yolo_labels, img_width, img_height
                        )

                        # Save augmented image
                        base_name = Path(os.path.basename(file_name)).stem
                        aug_img_name = f"{base_name}_aug{aug_idx}.jpg"
                        aug_img_path = os.path.join(img_dir, aug_img_name)
                        cv2.imwrite(aug_img_path, aug_image)

                        # Save augmented labels
                        aug_label_name = f"{base_name}_aug{aug_idx}.txt"
                        aug_label_path = os.path.join(lbl_dir, aug_label_name)
                        with open(aug_label_path, 'w') as f:
                            for label in aug_labels:
                                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

                        # Update augmentation counter
                        for c in img_rare_classes:
                            aug_counter[c] += 1

    # Create YOLO dataset YAML
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\nYOLO dataset YAML created: {yaml_path}")

    # Report augmentation results
    if augment_rare_classes and aug_counter:
        print(f"\n{'='*60}")
        print("AUGMENTATION RESULTS")
        print(f"{'='*60}")
        print(f"\n{'Class':<30} {'Augmented Samples'}")
        print("-"*60)
        for class_name in sorted(aug_counter.keys()):
            print(f"{class_name:<30} +{aug_counter[class_name]}")
        print(f"\nTotal augmented samples: {sum(aug_counter.values())}")

    # Save class mapping
    class_mapping = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_names)
    }

    mapping_path = os.path.join(output_dir, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)

    print(f"Class mapping saved: {mapping_path}")

    # Create dataset info
    info_path = os.path.join(output_dir, 'dataset_info.txt')
    with open(info_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("YOLO MULTI-CLASS DATASET (END-TO-END)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Train images: {len(train_image_ids)}\n")
        f.write(f"Val images: {len(val_image_ids)}\n")
        f.write(f"Total images: {len(all_image_ids)}\n")
        if augment_rare_classes and aug_counter:
            f.write(f"Augmented samples: {sum(aug_counter.values())}\n")
        f.write("\nPurpose:\n")
        f.write("  End-to-end detection + classification in single YOLO model\n")
        f.write("  For comparison with 2-stage cascade approach\n")
        if augment_rare_classes:
            f.write("  Rare classes augmented to balance training\n")
        f.write("\n")
        f.write("Classes:\n")
        for name in class_names:
            f.write(f"  - {name}\n")
        f.write("\n")
        f.write("="*60 + "\n")

    print(f"Dataset info saved: {info_path}")

    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE!")
    print("="*60)
    print(f"\nDataset location: {output_dir}")
    print(f"YAML config: {yaml_path}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/  ({len(train_image_ids)} images)")
    print(f"    │   └── val/    ({len(val_image_ids)} images)")
    print(f"    ├── labels/")
    print(f"    │   ├── train/  ({len(train_image_ids)} labels)")
    print(f"    │   └── val/    ({len(val_image_ids)} labels)")
    print(f"    ├── dataset.yaml")
    print(f"    ├── class_mapping.json")
    print(f"    └── dataset_info.txt")
    print(f"\nReady for YOLO end-to-end training!")
    print(f"\nCompare with cascade:")
    print(f"  Cascade: YOLO (1 class) → ArcFace (39 classes)")
    print(f"  This:    YOLO26 (39 classes) end-to-end")

    return output_dir

if __name__ == "__main__":
    # Configuration
    COCO_JSON = 'StudyCase/_annotations.coco.json'
    IMAGES_DIR = 'StudyCase/images'
    OUTPUT_DIR = 'yolo_multiclass_dataset'

    # Create YOLO multi-class dataset
    dataset_dir = create_yolo_multiclass_dataset(
        coco_json_path=COCO_JSON,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        train_split=0.8
    )

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n1. Train YOLO26 end-to-end (39 classes):")
    print(f"   bash train_yolo26_multiclass.sh")
    print("\n2. Or train YOLO26 cascade (1 class):")
    print(f"   bash train_yolo26_cascade.sh")
    print("\n3. Compare results:")
    print("   - End-to-end: Simpler, faster")
    print("   - Cascade: Better fine-grained accuracy")
