import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import yaml

def create_yolo_multiclass_dataset(
    coco_json_path: str,
    images_dir: str,
    output_dir: str = 'yolo_multiclass_dataset',
    train_split: float = 0.8
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

    # Split images into train/val (same split as cascade for fair comparison)
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

    # Create class name to index mapping (0-indexed for YOLO)
    class_names = sorted(categories.values())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    print(f"\nClass mapping (first 10):")
    for i, name in enumerate(class_names[:10]):
        print(f"  {i}: {name}")
    print(f"  ...")

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

                    # Get class index (0-indexed)
                    cat_name = categories[ann['category_id']]
                    class_idx = class_to_idx[cat_name]

                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

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
        f.write(f"Total images: {len(image_ids)}\n\n")
        f.write("Purpose:\n")
        f.write("  End-to-end detection + classification in single YOLO model\n")
        f.write("  For comparison with 2-stage cascade approach\n\n")
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
