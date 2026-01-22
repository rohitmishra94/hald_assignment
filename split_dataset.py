import json
import os
import shutil
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import random

def smart_train_test_split(
    coco_json_path: str,
    images_dir: str,
    output_dir: str = 'dataset_split',
    test_size: float = 0.2,
    min_samples_for_split: int = 5,
    stratify: bool = True,
    random_seed: int = 42
):
    """
    Smart split that handles rare classes intelligently

    Args:
        coco_json_path: Path to COCO format annotations
        images_dir: Path to images directory
        output_dir: Output directory for split dataset
        test_size: Fraction of data for test set (0.2 = 20%)
        min_samples_for_split: Classes with fewer samples go entirely to train
        stratify: Whether to maintain class distribution in splits
        random_seed: Random seed for reproducibility

    Strategy:
        - Classes with <min_samples_for_split: All samples → train
        - Classes with ≥min_samples_for_split: Stratified split
        - Ensures each class has representation in train set
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Loading annotations from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    train_dir = os.path.join(output_dir, 'train', 'images')
    test_dir = os.path.join(output_dir, 'test', 'images')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Group images by their classes
    image_to_classes = defaultdict(set)
    for ann in coco_data['annotations']:
        image_to_classes[ann['image_id']].add(ann['category_id'])

    # Count samples per class
    class_to_images = defaultdict(set)
    for img_id, class_ids in image_to_classes.items():
        for class_id in class_ids:
            class_to_images[class_id].add(img_id)

    # Categorize classes
    rare_classes = {}  # Classes that go entirely to train
    splittable_classes = {}  # Classes that can be split

    for class_id, img_ids in class_to_images.items():
        if len(img_ids) < min_samples_for_split:
            rare_classes[class_id] = img_ids
        else:
            splittable_classes[class_id] = img_ids

    # Create category mapping for logging
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    print(f"\nDataset Statistics:")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total classes: {len(categories)}")
    print(f"Rare classes (< {min_samples_for_split} samples): {len(rare_classes)}")
    print(f"Splittable classes (≥ {min_samples_for_split} samples): {len(splittable_classes)}")

    if rare_classes:
        print(f"\nRare classes (all → train):")
        for class_id in rare_classes:
            print(f"  - {categories[class_id]}: {len(rare_classes[class_id])} images")

    # Initialize train/test image sets
    train_images = set()
    test_images = set()

    # 1. All rare class images go to train
    for class_id, img_ids in rare_classes.items():
        train_images.update(img_ids)

    # 2. Split splittable classes
    if stratify:
        # Stratified split: maintain class distribution
        for class_id, img_ids in splittable_classes.items():
            img_list = list(img_ids)
            random.shuffle(img_list)

            n_test = max(1, int(len(img_list) * test_size))  # At least 1 in test
            n_train = len(img_list) - n_test

            # Split
            class_test = set(img_list[:n_test])
            class_train = set(img_list[n_test:])

            # Add to global sets (handle overlaps)
            # Images already in train (from rare classes) stay in train
            class_test = class_test - train_images

            test_images.update(class_test)
            train_images.update(class_train)

            print(f"\n{categories[class_id]}: {len(class_train)} train, {len(class_test)} test")
    else:
        # Random split without stratification
        remaining_images = set(range(1, len(coco_data['images']) + 1)) - train_images
        remaining_list = list(remaining_images)
        random.shuffle(remaining_list)

        n_test = int(len(remaining_list) * test_size)
        test_images = set(remaining_list[:n_test])
        train_images.update(remaining_list[n_test:])

    # Create image ID to filename mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Split annotations
    train_annotations = []
    test_annotations = []

    for ann in coco_data['annotations']:
        if ann['image_id'] in train_images:
            train_annotations.append(ann)
        elif ann['image_id'] in test_images:
            test_annotations.append(ann)

    # Create train COCO JSON
    train_coco = {
        'images': [img for img in coco_data['images'] if img['id'] in train_images],
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }

    # Create test COCO JSON
    test_coco = {
        'images': [img for img in coco_data['images'] if img['id'] in test_images],
        'annotations': test_annotations,
        'categories': coco_data['categories']
    }

    # Save COCO JSONs
    train_json_path = os.path.join(output_dir, 'train', 'annotations.json')
    test_json_path = os.path.join(output_dir, 'test', 'annotations.json')

    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)

    with open(test_json_path, 'w') as f:
        json.dump(test_coco, f, indent=2)

    print(f"\nSaved train annotations to {train_json_path}")
    print(f"Saved test annotations to {test_json_path}")

    # Copy images to respective directories
    print("\nCopying images...")

    for img_id in train_images:
        img_info = image_id_to_info[img_id]
        src = os.path.join(images_dir, img_info['file_name'])
        dst = os.path.join(train_dir, img_info['file_name'])

        if os.path.exists(src):
            shutil.copy2(src, dst)

    for img_id in test_images:
        img_info = image_id_to_info[img_id]
        src = os.path.join(images_dir, img_info['file_name'])
        dst = os.path.join(test_dir, img_info['file_name'])

        if os.path.exists(src):
            shutil.copy2(src, dst)

    print(f"\nCopied {len(train_images)} images to train/")
    print(f"Copied {len(test_images)} images to test/")

    # Generate split statistics
    stats = generate_split_statistics(train_coco, test_coco, categories)

    # Save statistics
    stats_path = os.path.join(output_dir, 'split_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write(stats)

    print(f"\nSplit statistics saved to {stats_path}")

    return {
        'train_images': len(train_images),
        'test_images': len(test_images),
        'train_annotations': len(train_annotations),
        'test_annotations': len(test_annotations),
        'train_json': train_json_path,
        'test_json': test_json_path
    }

def generate_split_statistics(
    train_coco: Dict,
    test_coco: Dict,
    categories: Dict
) -> str:
    """Generate detailed statistics about the split"""

    # Count per class
    train_class_counts = Counter()
    test_class_counts = Counter()

    for ann in train_coco['annotations']:
        train_class_counts[ann['category_id']] += 1

    for ann in test_coco['annotations']:
        test_class_counts[ann['category_id']] += 1

    stats = "=" * 60 + "\n"
    stats += "DATASET SPLIT STATISTICS\n"
    stats += "=" * 60 + "\n\n"

    stats += f"Train Images: {len(train_coco['images'])}\n"
    stats += f"Test Images: {len(test_coco['images'])}\n"
    stats += f"Total Images: {len(train_coco['images']) + len(test_coco['images'])}\n\n"

    stats += f"Train Annotations: {len(train_coco['annotations'])}\n"
    stats += f"Test Annotations: {len(test_coco['annotations'])}\n"
    stats += f"Total Annotations: {len(train_coco['annotations']) + len(test_coco['annotations'])}\n\n"

    stats += "=" * 60 + "\n"
    stats += "PER-CLASS DISTRIBUTION\n"
    stats += "=" * 60 + "\n"
    stats += f"{'Class':<30} {'Train':<10} {'Test':<10} {'Total':<10} {'Test %':<10}\n"
    stats += "-" * 60 + "\n"

    all_class_ids = sorted(set(train_class_counts.keys()) | set(test_class_counts.keys()))

    for class_id in all_class_ids:
        class_name = categories[class_id]
        train_count = train_class_counts[class_id]
        test_count = test_class_counts[class_id]
        total_count = train_count + test_count
        test_pct = (test_count / total_count * 100) if total_count > 0 else 0

        stats += f"{class_name:<30} {train_count:<10} {test_count:<10} {total_count:<10} {test_pct:<10.1f}\n"

    stats += "=" * 60 + "\n"

    return stats

def create_yolo_yaml(
    output_dir: str,
    dataset_name: str = 'plankton_dataset'
):
    """Create YOLO format dataset YAML configuration"""

    train_json_path = os.path.join(output_dir, 'train', 'annotations.json')

    with open(train_json_path, 'r') as f:
        coco_data = json.load(f)

    class_names = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]

    yaml_config = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(output_dir, f'{dataset_name}.yaml')

    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    print(f"\nYOLO dataset YAML created: {yaml_path}")

    return yaml_path

def augment_rare_classes(
    coco_json_path: str,
    images_dir: str,
    output_json_path: str,
    output_images_dir: str,
    min_samples: int = 10,
    augmentation_factor: int = 5
):
    """
    Apply augmentation to rare classes to increase sample size
    Uses strong augmentation to create synthetic samples
    """
    import cv2
    import albumentations as A

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Count samples per class
    class_counts = Counter()
    for ann in coco_data['annotations']:
        class_counts[ann['category_id']] += 1

    # Identify rare classes
    rare_classes = {cls_id for cls_id, count in class_counts.items() if count < min_samples}

    if not rare_classes:
        print("No rare classes found!")
        return

    print(f"Augmenting {len(rare_classes)} rare classes...")

    # Augmentation pipeline
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ElasticTransform(alpha=20, sigma=5, p=0.2)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    os.makedirs(output_images_dir, exist_ok=True)

    new_images = []
    new_annotations = []
    next_img_id = max(img['id'] for img in coco_data['images']) + 1
    next_ann_id = max(ann['id'] for ann in coco_data['annotations']) + 1

    # Copy original data
    for img in coco_data['images']:
        src = os.path.join(images_dir, img['file_name'])
        dst = os.path.join(output_images_dir, img['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)

    new_images.extend(coco_data['images'])
    new_annotations.extend(coco_data['annotations'])

    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # Augment images containing rare classes
    for img_info in coco_data['images']:
        img_anns = img_to_anns[img_info['id']]

        # Check if image contains rare classes
        has_rare = any(ann['category_id'] in rare_classes for ann in img_anns)

        if not has_rare:
            continue

        # Load image
        img_path = os.path.join(images_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create augmented versions
        for aug_idx in range(augmentation_factor):
            # Prepare bboxes and labels
            bboxes = [ann['bbox'] for ann in img_anns]
            category_ids = [ann['category_id'] for ann in img_anns]

            # Apply augmentation
            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_category_ids = transformed['category_ids']

            # Save augmented image
            base_name = Path(img_info['file_name']).stem
            ext = Path(img_info['file_name']).suffix
            aug_filename = f"{base_name}_aug{aug_idx}{ext}"
            aug_path = os.path.join(output_images_dir, aug_filename)

            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(aug_path, aug_image_bgr)

            # Create new image entry
            new_img_info = {
                'id': next_img_id,
                'file_name': aug_filename,
                'height': img_info['height'],
                'width': img_info['width']
            }
            new_images.append(new_img_info)

            # Create new annotation entries
            for bbox, cat_id in zip(aug_bboxes, aug_category_ids):
                new_ann = {
                    'id': next_ann_id,
                    'image_id': next_img_id,
                    'category_id': cat_id,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0
                }
                new_annotations.append(new_ann)
                next_ann_id += 1

            next_img_id += 1

    # Create new COCO JSON
    augmented_coco = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco_data['categories']
    }

    with open(output_json_path, 'w') as f:
        json.dump(augmented_coco, f, indent=2)

    print(f"\nAugmented dataset created:")
    print(f"Original images: {len(coco_data['images'])}")
    print(f"Augmented images: {len(new_images)}")
    print(f"Original annotations: {len(coco_data['annotations'])}")
    print(f"Augmented annotations: {len(new_annotations)}")

if __name__ == "__main__":
    # Configuration
    COCO_JSON = 'annotation.coco.json'
    IMAGES_DIR = 'workspace/some_exp/genus/hald_assignment/StudyCase/images'
    OUTPUT_DIR = 'dataset_split'

    # Option 1: Smart split (recommended)
    print("=" * 60)
    print("SMART TRAIN/TEST SPLIT")
    print("=" * 60)

    result = smart_train_test_split(
        coco_json_path=COCO_JSON,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        test_size=0.2,  # 20% for test
        min_samples_for_split=5,  # Classes with <5 samples go to train only
        stratify=True,
        random_seed=42
    )

    # Create YOLO YAML
    yaml_path = create_yolo_yaml(OUTPUT_DIR, 'plankton_dataset')

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE!")
    print("=" * 60)
    print(f"Train images: {result['train_images']}")
    print(f"Test images: {result['test_images']}")
    print(f"Train annotations: {result['train_annotations']}")
    print(f"Test annotations: {result['test_annotations']}")
    print(f"\nUse this YAML for training: {yaml_path}")

    # Option 2: Augment rare classes first (optional)
    # Uncomment if you want to augment rare classes before splitting
    """
    print("\n" + "=" * 60)
    print("AUGMENTING RARE CLASSES")
    print("=" * 60)

    augment_rare_classes(
        coco_json_path=COCO_JSON,
        images_dir=IMAGES_DIR,
        output_json_path='augmented_annotations.json',
        output_images_dir='augmented_images',
        min_samples=10,
        augmentation_factor=5
    )

    # Then split the augmented dataset
    result = smart_train_test_split(
        coco_json_path='augmented_annotations.json',
        images_dir='augmented_images',
        output_dir='dataset_split_augmented',
        test_size=0.2,
        min_samples_for_split=5,
        stratify=True,
        random_seed=42
    )
    """
