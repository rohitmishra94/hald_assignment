"""
Generate confusion matrix with actual counts for ArcFace validation set
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
from model_arc import EmbeddingModel, SubCenterArcFaceHead

def generate_confusion_matrix_with_numbers(
    model_path: str,
    dataset_path: str,
    class_mapping_path: str,
    output_dir: str = '../arcface_models',
    device: str = 'cuda'
):
    """
    Generate confusion matrix with actual prediction counts

    Args:
        model_path: Path to best_model.pth
        dataset_path: Path to test dataset
        class_mapping_path: Path to class_mapping.json
        output_dir: Where to save outputs
        device: 'cuda' or 'cpu'
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load class mapping
    print(f"Loading class mapping from {class_mapping_path}...")
    with open(class_mapping_path, 'r') as f:
        class_info = json.load(f)

    num_classes = class_info['num_classes']
    idx_to_class = {int(k): v for k, v in class_info['idx_to_class'].items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    print(f"Number of classes: {num_classes}")

    # Load model (backbone only, matching train_arc.py structure)
    print(f"Loading model from {model_path}...")

    # Initialize backbone
    backbone = EmbeddingModel(
        embedding_size=512,
        pretrained=False,
        backbone='resnet50'
    )

    # Load weights (train_arc.py saves only backbone.state_dict())
    backbone.load_state_dict(torch.load(model_path, map_location=device))
    backbone = backbone.to(device)
    backbone.eval()

    # Initialize metric head (needed for inference)
    # Using same config as training: k=5, s=30.0, m=0.35
    metric_head = SubCenterArcFaceHead(512, num_classes, k=5, s=30.0, m=0.35).to(device)
    metric_head.eval()

    # Load validation dataset
    print(f"Loading validation dataset from {dataset_path}...")
    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(dataset_path, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    print(f"Validation samples: {len(val_dataset)}")

    # Get predictions
    print("Running inference on validation set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get embeddings from backbone
            embeddings = backbone(images)

            # Pass through SubCenterArcFace head for logits
            logits = metric_head(embeddings, labels)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"Total predictions: {len(all_preds)}")

    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    # Save confusion matrix as CSV with numbers
    cm_csv_path = os.path.join(output_dir, 'confusion_matrix_counts.csv')
    print(f"\nSaving confusion matrix counts to {cm_csv_path}...")

    import pandas as pd
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_csv_path)

    # Also save as text file for easy viewing
    cm_txt_path = os.path.join(output_dir, 'confusion_matrix_counts.txt')
    print(f"Saving confusion matrix text to {cm_txt_path}...")

    with open(cm_txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("CONFUSION MATRIX (Actual Counts)\n")
        f.write("="*100 + "\n\n")
        f.write("Rows = True Labels, Columns = Predicted Labels\n\n")

        # Write header
        f.write(f"{'True \\ Predicted':<30}")
        for class_name in class_names:
            f.write(f"{class_name[:15]:<17}")
        f.write("\n")
        f.write("-"*100 + "\n")

        # Write matrix
        for i, true_class in enumerate(class_names):
            f.write(f"{true_class:<30}")
            for j in range(num_classes):
                count = cm[i, j]
                if count > 0:
                    f.write(f"{count:<17}")
                else:
                    f.write(f"{'.':<17}")
            f.write("\n")

        f.write("\n" + "="*100 + "\n")
        f.write("DIAGONAL (Correct Predictions):\n")
        f.write("="*100 + "\n")
        for i, class_name in enumerate(class_names):
            correct = cm[i, i]
            total = np.sum(cm[i, :])
            if total > 0:
                acc = 100 * correct / total
                f.write(f"{class_name:<30} {correct:>5}/{total:<5} ({acc:>6.2f}%)\n")

        f.write("\n" + "="*100 + "\n")
        f.write("MISCLASSIFICATIONS (Off-diagonal):\n")
        f.write("="*100 + "\n")

        misclassifications = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm[i, j] > 0:
                    misclassifications.append((
                        class_names[i],
                        class_names[j],
                        cm[i, j]
                    ))

        misclassifications.sort(key=lambda x: x[2], reverse=True)

        if misclassifications:
            f.write(f"{'True Class':<30} {'Predicted As':<30} {'Count':<10}\n")
            f.write("-"*70 + "\n")
            for true_class, pred_class, count in misclassifications:
                f.write(f"{true_class:<30} {pred_class:<30} {count:<10}\n")
        else:
            f.write("No misclassifications! Perfect accuracy!\n")

    # Generate annotated heatmap with numbers
    print(f"\nGenerating confusion matrix heatmap...")

    # Create normalized version for color intensity
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Plot with actual counts annotated
    plt.figure(figsize=(20, 18))

    # Use normalized for colors, but annotate with actual counts
    sns.heatmap(
        cm_normalized,
        annot=cm,  # Show actual counts
        fmt='d',   # Integer format
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'},
        square=False,
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title('Confusion Matrix with Counts\n(Color = Normalized, Numbers = Actual Counts)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, 'confusion_matrix_with_numbers.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {heatmap_path}")
    plt.close()

    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Calculate and print summary statistics
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    accuracy = 100 * total_correct / total_samples

    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {total_correct}")
    print(f"Misclassifications: {total_samples - total_correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nTotal misclassification pairs: {len(misclassifications)}")

    if misclassifications:
        print(f"\nTop 5 most common misclassifications:")
        for i, (true_class, pred_class, count) in enumerate(misclassifications[:5], 1):
            print(f"  {i}. {true_class} → {pred_class}: {count} times")

    print("\n" + "="*100)
    print("FILES SAVED:")
    print("="*100)
    print(f"  • {cm_csv_path}")
    print(f"  • {cm_txt_path}")
    print(f"  • {heatmap_path}")
    print("="*100)

    return cm, class_names


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = '../arcface_models/best_model.pth'
    DATASET_PATH = '../arcface_dataset/test'
    CLASS_MAPPING_PATH = '../arcface_dataset/class_mapping.json'
    OUTPUT_DIR = '../arcface_models'

    print("="*100)
    print("GENERATING CONFUSION MATRIX WITH COUNTS")
    print("="*100)
    print()

    cm, class_names = generate_confusion_matrix_with_numbers(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        class_mapping_path=CLASS_MAPPING_PATH,
        output_dir=OUTPUT_DIR,
        device='cuda'
    )

    print("\nDone!")
