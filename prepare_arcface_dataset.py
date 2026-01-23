import json
import os
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm
import random

def extract_cropped_objects_for_arcface(
    coco_json_path: str,
    images_dir: str,
    output_dir: str = 'arcface_dataset',
    target_size: Tuple[int, int] = (128, 128),
    min_object_size: int = 20,
    padding_ratio: float = 0.1,
    augment_rare_classes: bool = True,
    min_samples_for_augmentation: int = 50,
    test_split: float = 0.2,
    random_seed: int = 42
):
    """
    Convert COCO format dataset to cropped images for ArcFace training

    Args:
        coco_json_path: Path to COCO annotations
        images_dir: Path to source images
        output_dir: Output directory for cropped images
        target_size: Resize cropped objects to this size (128x128 for ArcFace)
        min_object_size: Minimum bbox size to include
        padding_ratio: Add padding around bbox (0.1 = 10% padding)
        augment_rare_classes: Whether to augment rare classes
        min_samples_for_augmentation: Classes with fewer samples get augmented
        test_split: Fraction for test set
        random_seed: Random seed for reproducibility
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Loading annotations from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Create class directories
    for cat_name in categories.values():
        os.makedirs(os.path.join(train_dir, cat_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cat_name), exist_ok=True)

    # Group annotations by category
    category_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        bbox = ann['bbox']
        # Keep all objects if min_object_size is 0, otherwise filter
        if min_object_size == 0 or (bbox[2] >= min_object_size and bbox[3] >= min_object_size):
            category_annotations[ann['category_id']].append(ann)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total classes: {len(categories)}")
    print(f"Total annotations: {len(coco_data['annotations'])}")

    class_counts = {cat_id: len(anns) for cat_id, anns in category_annotations.items()}
    print("\nPer-class counts:")
    for cat_id, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {categories[cat_id]}: {count}")

    # Create image ID to info mapping
    image_map = {img['id']: img for img in coco_data['images']}

    # Process each category
    train_count = 0
    test_count = 0

    for cat_id, annotations in tqdm(category_annotations.items(), desc="Processing classes"):
        cat_name = categories[cat_id]

        # Shuffle annotations
        random.shuffle(annotations)

        # Smart split: ensure at least 1 in train, rest goes to test if enough samples
        if len(annotations) == 1:
            # Single sample: keep in train, augment later
            train_annotations = annotations
            test_annotations = []
        elif len(annotations) == 2:
            # Two samples: 1 train, 1 test
            train_annotations = annotations[:1]
            test_annotations = annotations[1:]
        else:
            # Multiple samples: ensure at least 1 in train, rest split by ratio
            n_test = max(1, int(len(annotations) * test_split))
            test_annotations = annotations[:n_test]
            train_annotations = annotations[n_test:]

        # Process train annotations
        for idx, ann in enumerate(train_annotations):
            img_info = image_map[ann['image_id']]

            # Handle file_name that may include 'images/' prefix
            file_name = img_info['file_name']
            if file_name.startswith('images/'):
                file_name = file_name.replace('images/', '', 1)

            img_path = os.path.join(images_dir, file_name)

            if not os.path.exists(img_path):
                continue

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Extract and save cropped object
            cropped = extract_and_resize_object(
                image, ann['bbox'], target_size, padding_ratio
            )

            if cropped is not None:
                output_path = os.path.join(
                    train_dir,
                    cat_name,
                    f"{cat_name}_{ann['id']}.jpg"
                )
                cv2.imwrite(output_path, cropped)
                train_count += 1

        # Process test annotations
        for idx, ann in enumerate(test_annotations):
            img_info = image_map[ann['image_id']]

            # Handle file_name that may include 'images/' prefix
            file_name = img_info['file_name']
            if file_name.startswith('images/'):
                file_name = file_name.replace('images/', '', 1)

            img_path = os.path.join(images_dir, file_name)

            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            cropped = extract_and_resize_object(
                image, ann['bbox'], target_size, padding_ratio
            )

            if cropped is not None:
                output_path = os.path.join(
                    test_dir,
                    cat_name,
                    f"{cat_name}_{ann['id']}.jpg"
                )
                cv2.imwrite(output_path, cropped)
                test_count += 1

    print(f"\nExtraction complete!")
    print(f"Train images: {train_count}")
    print(f"Test images: {test_count}")

    # Augment rare classes if requested
    if augment_rare_classes:
        print("\nAugmenting rare classes...")
        augment_rare_classes_for_arcface(
            train_dir,
            min_samples=min_samples_for_augmentation,
            target_samples=min_samples_for_augmentation * 2
        )

    # Generate dataset info
    generate_dataset_info(output_dir, train_dir, test_dir)

    return output_dir

def extract_and_resize_object(
    image: np.ndarray,
    bbox: List[float],
    target_size: Tuple[int, int],
    padding_ratio: float = 0.1
) -> np.ndarray:
    """
    Extract object from image with padding and resize
    """
    x, y, w, h = bbox

    # Add padding
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(image.shape[1], int(x + w + pad_w))
    y2 = min(image.shape[0], int(y + h + pad_h))

    # Crop
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return None

    # Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)

    return resized

def augment_rare_classes_for_arcface(
    train_dir: str,
    min_samples: int = 50,
    target_samples: int = 100
):
    """
    Augment rare classes using strong augmentation
    """
    import albumentations as A

    # Augmentation pipeline for metric learning
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.3),
        A.Affine(
            translate_percent=0.05,
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.3
        ),
        A.GaussNoise(var_limit=(0.001, 0.003), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(4, 8),
            hole_width_range=(4, 8),
            p=0.2
        )
    ])

    # Find rare classes
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    for class_name in tqdm(class_dirs, desc="Augmenting rare classes"):
        class_dir = os.path.join(train_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]

        n_samples = len(images)

        if n_samples >= min_samples:
            continue

        print(f"\n  Augmenting {class_name}: {n_samples} → {target_samples}")

        # Calculate how many augmented versions per image
        n_augmentations = (target_samples - n_samples) // n_samples + 1

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            # Generate augmented versions
            for aug_idx in range(n_augmentations):
                augmented = transform(image=image)['image']

                # Save augmented image
                base_name = Path(img_name).stem
                aug_name = f"{base_name}_aug{aug_idx}.jpg"
                aug_path = os.path.join(class_dir, aug_name)
                cv2.imwrite(aug_path, augmented)

def generate_dataset_info(output_dir: str, train_dir: str, test_dir: str):
    """
    Generate dataset information file
    """
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    info = {
        'num_classes': len(train_classes),
        'classes': train_classes,
        'train_counts': {},
        'test_counts': {},
        'total_train': 0,
        'total_test': 0
    }

    for class_name in train_classes:
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        train_count = len([f for f in os.listdir(train_class_dir) if f.endswith(('.jpg', '.png'))])
        test_count = len([f for f in os.listdir(test_class_dir) if f.endswith(('.jpg', '.png'))])

        info['train_counts'][class_name] = train_count
        info['test_counts'][class_name] = test_count
        info['total_train'] += train_count
        info['total_test'] += test_count

    # Save as JSON
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    # Save as text
    txt_path = os.path.join(output_dir, 'dataset_info.txt')
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ARCFACE DATASET INFORMATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of classes: {info['num_classes']}\n")
        f.write(f"Total train images: {info['total_train']}\n")
        f.write(f"Total test images: {info['total_test']}\n\n")
        f.write("="*60 + "\n")
        f.write("PER-CLASS COUNTS\n")
        f.write("="*60 + "\n")
        f.write(f"{'Class':<30} {'Train':<10} {'Test':<10} {'Total':<10}\n")
        f.write("-"*60 + "\n")

        for class_name in sorted(info['train_counts'].keys()):
            train_c = info['train_counts'][class_name]
            test_c = info['test_counts'][class_name]
            total_c = train_c + test_c
            f.write(f"{class_name:<30} {train_c:<10} {test_c:<10} {total_c:<10}\n")

        f.write("="*60 + "\n")

    print(f"\nDataset info saved to:")
    print(f"  {info_path}")
    print(f"  {txt_path}")

    return info

def create_class_mapping(output_dir: str):
    """
    Create class to index mapping for training
    """
    info_path = os.path.join(output_dir, 'dataset_info.json')

    with open(info_path, 'r') as f:
        info = json.load(f)

    class_to_idx = {class_name: idx for idx, class_name in enumerate(info['classes'])}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    mapping = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_to_idx)
    }

    mapping_path = os.path.join(output_dir, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\nClass mapping saved to: {mapping_path}")

    return mapping

if __name__ == "__main__":
    # Configuration
    COCO_JSON = 'workspace/some_exp/genus/hald_assignment/StudyCase/_annotations.coco.json'
    IMAGES_DIR = 'workspace/some_exp/genus/hald_assignment/StudyCase/images'
    OUTPUT_DIR = 'arcface_dataset'

    print("="*60)
    print("PREPARING ARCFACE DATASET FROM COCO")
    print("="*60)

    # Extract and prepare dataset
    output_dir = extract_cropped_objects_for_arcface(
        coco_json_path=COCO_JSON,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        target_size=(128, 128),
        min_object_size=0,  # Keep all objects (ground truth annotations)
        padding_ratio=0.1,
        augment_rare_classes=True,
        min_samples_for_augmentation=50,
        test_split=0.2,
        random_seed=42
    )

    # Create class mapping
    mapping = create_class_mapping(output_dir)

    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nDataset location: {output_dir}")
    print(f"Number of classes: {mapping['num_classes']}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── class1/")
    print(f"    │   ├── class2/")
    print(f"    │   └── ...")
    print(f"    ├── test/")
    print(f"    │   ├── class1/")
    print(f"    │   ├── class2/")
    print(f"    │   └── ...")
    print(f"    ├── dataset_info.json")
    print(f"    ├── dataset_info.txt")
    print(f"    └── class_mapping.json")
    print(f"\nUse this dataset to train your ArcFace model!")
