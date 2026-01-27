import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import yaml

def create_superclass_coco(
    coco_json_path: str,
    output_json_path: str,
    superclass_name: str = 'plankton'
):
    """
    Convert multi-class COCO to single super-class for YOLO detection

    This is for Stage 1 of the cascade: detect ALL plankton objects
    regardless of species. Stage 2 (ArcFace) will handle identification.

    Args:
        coco_json_path: Original COCO annotations
        output_json_path: Output path for super-class COCO
        superclass_name: Name for the super-class (e.g., 'plankton')
    """

    print(f"Loading COCO annotations from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Get original categories for reference
    original_categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"\nOriginal dataset:")
    print(f"  Classes: {len(original_categories)}")
    print(f"  Annotations: {len(coco_data['annotations'])}")

    # Create single super-class category
    superclass_coco = {
        'images': coco_data['images'],
        'categories': [
            {
                'id': 1,
                'name': superclass_name,
                'supercategory': superclass_name
            }
        ],
        'annotations': []
    }

    # Convert all annotations to super-class
    for ann in coco_data['annotations']:
        new_ann = ann.copy()
        new_ann['category_id'] = 1  # All objects become class 1
        superclass_coco['annotations'].append(new_ann)

    # Save super-class COCO
    with open(output_json_path, 'w') as f:
        json.dump(superclass_coco, f, indent=2)

    print(f"\nSuper-class dataset created:")
    print(f"  Classes: 1 ({superclass_name})")
    print(f"  Annotations: {len(superclass_coco['annotations'])}")
    print(f"  Saved to: {output_json_path}")

    # Create mapping file for reference
    mapping_path = output_json_path.replace('.json', '_mapping.json')
    mapping = {
        'superclass': superclass_name,
        'original_classes': original_categories,
        'note': 'All original classes mapped to single super-class for detection'
    }

    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"  Mapping saved to: {mapping_path}")

    return superclass_coco

def create_yolo_superclass_dataset(
    coco_json_path: str,
    images_dir: str,
    output_dir: str = 'yolo_superclass_dataset',
    superclass_name: str = 'plankton',
    train_split: float = 0.8
):
    """
    Create complete YOLO dataset with single super-class

    Creates directory structure:
    yolo_superclass_dataset/
      ├── images/
      │   ├── train/
      │   └── val/
      ├── labels/
      │   ├── train/
      │   └── val/
      ├── dataset.yaml
      └── superclass_annotations.json
    """

    print("="*60)
    print("CREATING YOLO SUPER-CLASS DATASET")
    print("="*60)

    # Create output directories
    images_train_dir = os.path.join(output_dir, 'images', 'train')
    images_val_dir = os.path.join(output_dir, 'images', 'val')
    labels_train_dir = os.path.join(output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(output_dir, 'labels', 'val')

    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Create super-class COCO
    superclass_json = os.path.join(output_dir, 'superclass_annotations.json')
    superclass_coco = create_superclass_coco(
        coco_json_path, superclass_json, superclass_name
    )

    # Load annotations
    with open(superclass_json, 'r') as f:
        coco_data = json.load(f)

    # Create image ID to info mapping
    image_map = {img['id']: img for img in coco_data['images']}

    # Group annotations by image
    from collections import defaultdict
    image_to_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_to_annotations[ann['image_id']].append(ann)

    # Split images into train/val
    import random
    random.seed(42)

    image_ids = list(image_to_annotations.keys())
    random.shuffle(image_ids)

    n_train = int(len(image_ids) * train_split)
    train_image_ids = set(image_ids[:n_train])
    val_image_ids = set(image_ids[n_train:])

    print(f"\nSplitting dataset:")
    print(f"  Train images: {len(train_image_ids)}")
    print(f"  Val images: {len(val_image_ids)}")

    # Process each image
    from tqdm import tqdm

    for split_name, split_ids, img_dir, lbl_dir in [
        ('train', train_image_ids, images_train_dir, labels_train_dir),
        ('val', val_image_ids, images_val_dir, labels_val_dir)
    ]:
        print(f"\nProcessing {split_name} split...")

        for img_id in tqdm(split_ids):
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

            with open(label_path, 'w') as f:
                for ann in annotations:
                    bbox = ann['bbox']  # [x, y, width, height] in pixels

                    # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width_norm = bbox[2] / img_width
                    height_norm = bbox[3] / img_height

                    # Class ID is 0 (single class)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

    # Create YOLO dataset YAML
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': [superclass_name]
    }

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\nYOLO dataset YAML created: {yaml_path}")

    # Create dataset info
    info_path = os.path.join(output_dir, 'dataset_info.txt')
    with open(info_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("YOLO SUPER-CLASS DATASET\n")
        f.write("="*60 + "\n\n")
        f.write(f"Super-class: {superclass_name}\n")
        f.write(f"Train images: {len(train_image_ids)}\n")
        f.write(f"Val images: {len(val_image_ids)}\n")
        f.write(f"Total images: {len(image_ids)}\n\n")
        f.write("Purpose:\n")
        f.write("  Stage 1: Detect ALL plankton objects (high recall)\n")
        f.write("  Stage 2: ArcFace will identify specific species\n\n")
        f.write("Training notes:\n")
        f.write("  - Optimize for HIGH RECALL (detect all objects)\n")
        f.write("  - False positives acceptable (will be filtered by ArcFace)\n")
        f.write("  - Use lower confidence threshold during inference\n")
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
    print(f"    ├── superclass_annotations.json")
    print(f"    └── dataset_info.txt")
    print(f"\nReady for YOLO training!")
    print(f"Run: python train_yolo.py --data {yaml_path}")

    return output_dir

if __name__ == "__main__":
    # Configuration
    COCO_JSON = 'StudyCase/_annotations.coco.json'
    IMAGES_DIR = 'StudyCase/images'
    OUTPUT_DIR = 'yolo_superclass_dataset'

    # Create YOLO super-class dataset
    dataset_dir = create_yolo_superclass_dataset(
        coco_json_path=COCO_JSON,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        superclass_name='plankton',
        train_split=0.8
    )

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n1. Train YOLO detector (Stage 1):")
    print(f"   python train_yolo.py --data {dataset_dir}/dataset.yaml")
    print("\n2. Prepare ArcFace dataset (Stage 2):")
    print("   python prepare_arcface_dataset.py")
    print("\n3. Train ArcFace identifier:")
    print("   python train_arc.py")
    print("\n4. Run cascade inference:")
    print("   python cascade_inference.py --image <path>")
