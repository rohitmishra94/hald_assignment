import os
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from sklearn.utils.class_weight import compute_class_weight
import json
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset, DataLoader

def create_yolo_dataset_yaml(
    coco_json_path: str,
    images_dir: str,
    output_yaml: str = 'plankton_dataset.yaml'
):
    """Create YOLO format dataset configuration"""
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Get class names
    classes = {cat['id']: cat['name'] for cat in coco_data['categories']}
    class_names = [classes[i] for i in sorted(classes.keys())]

    # Create YAML config
    dataset_config = {
        'path': os.path.dirname(images_dir),
        'train': 'images',
        'val': 'images',  # You should split your data properly
        'nc': len(class_names),
        'names': class_names
    }

    with open(output_yaml, 'w') as f:
        yaml.dump(dataset_config, f)

    return output_yaml, class_names

def calculate_class_weights(coco_json_path: str):
    """Calculate class weights for handling imbalance"""
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Count instances per class
    class_counts = Counter()
    for ann in coco_data['annotations']:
        class_counts[ann['category_id']] += 1

    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)

    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (n_classes * count)

    # Normalize weights
    max_weight = max(weights.values())
    weights = {k: v/max_weight * 2.0 for k, v in weights.items()}  # Scale to max 2.0

    return weights

def get_augmentation_pipeline():
    """
    Strong augmentation pipeline for handling intra-class variance
    and improving generalization
    """
    return A.Compose([
        # Spatial augmentations for intra-class variance
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            p=0.5
        ),

        # Lighting augmentations for microscopy variance
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=0.3
        ),
        A.RandomGamma(
            gamma_limit=(70, 130),
            p=0.3
        ),

        # Color augmentations
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),

        # Noise and blur for robustness
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Microscopy-specific augmentations
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.2
        ),

        # Elastic deformation for biological objects
        A.ElasticTransform(
            alpha=20,
            sigma=5,
            alpha_affine=0,
            p=0.2
        )
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def train_yolo_v10(
    dataset_yaml: str,
    class_weights: dict = None,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    model_name: str = 'yolov10n',  # n/s/m/l/x variants
    patience: int = 20,
    device: str = 'cuda',
    project_name: str = 'plankton_yolo',
    use_wandb: bool = False
):
    """
    Train YOLOv10 with optimizations for plankton detection
    """

    # Initialize wandb if requested
    if use_wandb:
        import wandb
        wandb.init(project=project_name, name=f"{model_name}_{epochs}ep")

    # Load model
    if model_name == 'yolov10n':
        model = YOLO('yolov10n.pt')  # Nano version
    elif model_name == 'yolov10s':
        model = YOLO('yolov10s.pt')  # Small version
    elif model_name == 'yolov10m':
        model = YOLO('yolov10m.pt')  # Medium version
    elif model_name == 'yolov10l':
        model = YOLO('yolov10l.pt')  # Large version
    else:
        model = YOLO('yolov10x.pt')  # Extra large version

    # Training arguments optimized for your use case
    train_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'patience': patience,
        'save': True,
        'project': project_name,
        'name': f'{model_name}_plankton',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,   # Final learning rate factor
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # Loss weights - crucial for your requirements
        'box': 7.5,     # Increased for better localization
        'cls': 0.5,     # Class loss weight
        'dfl': 1.5,     # Distribution focal loss

        # NMS parameters - important for reducing ghost detections
        'conf': 0.25,   # Confidence threshold
        'iou': 0.45,    # IoU threshold for NMS
        'max_det': 300, # Maximum detections per image

        # Augmentation settings
        'hsv_h': 0.015, # Hue augmentation
        'hsv_s': 0.7,   # Saturation augmentation
        'hsv_v': 0.4,   # Value augmentation
        'degrees': 45,   # Rotation augmentation
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.5,  # Vertical flip
        'fliplr': 0.5,  # Horizontal flip
        'mosaic': 1.0,  # Mosaic augmentation
        'mixup': 0.2,   # MixUp augmentation for better generalization
        'copy_paste': 0.3,  # Copy-paste augmentation for rare classes

        # Advanced settings
        'label_smoothing': 0.1,  # Helps with inter-class similarity
        'nbs': 64,  # Nominal batch size for gradient accumulation
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.1,  # Add dropout for better generalization

        # Validation settings
        'val': True,
        'split': 0.2,  # Validation split
        'save_json': True,
        'save_hybrid': True,

        # Hardware optimization
        'workers': 8,
        'cache': True,  # Cache images for faster training
        'amp': True,    # Automatic mixed precision

        # Logging
        'plots': True,
        'verbose': True
    }

    # Add class weights if provided
    if class_weights:
        # Convert to list ordered by class id
        weights_list = [class_weights.get(i, 1.0) for i in range(len(class_weights))]
        train_args['cls_pw'] = weights_list

    # Custom callback for handling difficult cases
    def on_train_epoch_end(trainer):
        """Custom callback to adjust training based on metrics"""
        metrics = trainer.metrics

        # Adjust confidence threshold if too many false positives
        if 'precision' in metrics and metrics['precision'] < 0.7:
            trainer.args.conf = min(trainer.args.conf + 0.05, 0.5)

        # Log to wandb if enabled
        if use_wandb:
            import wandb
            wandb.log(metrics)

    # Add callback
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # Train the model
    results = model.train(**train_args)

    # Validate the model
    val_results = model.val()

    # Export best model
    best_model_path = f"{project_name}/{model_name}_plankton/weights/best.pt"

    # Export to different formats
    model = YOLO(best_model_path)

    # Export to ONNX for deployment
    model.export(format='onnx', simplify=True, dynamic=True)

    # Export to TensorRT for faster inference (if CUDA available)
    if torch.cuda.is_available():
        try:
            model.export(format='engine', half=True)  # FP16 for speed
        except:
            print("TensorRT export failed, continuing with ONNX")

    print(f"\nTraining complete! Best model saved at: {best_model_path}")
    print(f"Validation mAP@0.5: {val_results.box.map50:.4f}")
    print(f"Validation mAP@0.5:0.95: {val_results.box.map:.4f}")

    return best_model_path, results, val_results

def create_ensemble_config(models: list, output_path: str = 'ensemble_config.yaml'):
    """
    Create configuration for model ensemble to reduce false positives
    """
    ensemble_config = {
        'models': models,
        'weights': [1.0] * len(models),  # Equal weights, can be optimized
        'iou_threshold': 0.5,
        'conf_threshold': 0.3,
        'voting': 'weighted'  # 'unanimous', 'majority', 'weighted'
    }

    with open(output_path, 'w') as f:
        yaml.dump(ensemble_config, f)

    return output_path

if __name__ == "__main__":
    # Configuration
    COCO_JSON = 'annotation.coco.json'
    IMAGES_DIR = 'workspace/some_exp/genus/hald_assignment/StudyCase/images'

    # Create YOLO dataset configuration
    dataset_yaml, class_names = create_yolo_dataset_yaml(
        COCO_JSON,
        IMAGES_DIR,
        'plankton_dataset.yaml'
    )

    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(COCO_JSON)
    print(f"Class weights calculated: {class_weights}")

    # Train multiple models for ensemble (optional but recommended)
    models_to_train = [
        {'name': 'yolov10n', 'epochs': 100, 'batch_size': 32},  # Fast, less accurate
        {'name': 'yolov10s', 'epochs': 150, 'batch_size': 16},  # Balanced
        # {'name': 'yolov10m', 'epochs': 200, 'batch_size': 8},  # More accurate
    ]

    trained_models = []

    for config in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {config['name']}...")
        print('='*50)

        model_path, results, val_results = train_yolo_v10(
            dataset_yaml=dataset_yaml,
            class_weights=class_weights,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            img_size=640,
            model_name=config['name'],
            patience=20,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            project_name='plankton_detection',
            use_wandb=False  # Set to True if you have wandb account
        )

        trained_models.append(model_path)

    # Create ensemble configuration
    if len(trained_models) > 1:
        ensemble_config = create_ensemble_config(trained_models)
        print(f"\nEnsemble configuration saved to: {ensemble_config}")

    print("\n✅ Training complete! Use inference.py for predictions.")
