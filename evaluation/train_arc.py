import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_arc import EmbeddingModel, ArcFaceHead, SubCenterArcFaceHead
import os
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
parser = argparse.ArgumentParser(description="Train ArcFace Model")
parser.add_argument('--dataset', type=str, default='../arcface_dataset',
                   help='Path to ArcFace dataset directory')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--embedding-size', type=int, default=512)
parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50'],
                   help='Backbone architecture')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--output', type=str, default='../arcface_models',
                   help='Output directory for models')

args = parser.parse_args()

data_dir = os.path.join(args.dataset, 'train')
BATCH_SIZE = args.batch_size
LR = args.lr
EPOCHS = args.epochs
EMBEDDING_SIZE = args.embedding_size
DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

# --- Transforms (Critical for Variance Handling) ---
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),  # Increased from 10 to 45 degrees
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
    ),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
])

val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Data Loading ---
print(f"Loading dataset from: {data_dir}")
train_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Load validation set
val_dir = os.path.join(args.dataset, 'test')
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_classes = len(train_dataset.classes)

print(f"\n{'='*60}")
print(f"ARCFACE TRAINING CONFIGURATION")
print(f"{'='*60}")
print(f"Train dataset: {data_dir}")
print(f"Val dataset: {val_dir}")
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LR}")
print(f"Embedding size: {EMBEDDING_SIZE}")
print(f"Device: {DEVICE}")
print(f"{'='*60}\n")

# --- Initialize Models ---
backbone = EmbeddingModel(
    embedding_size=EMBEDDING_SIZE,
    pretrained=True,
    backbone=args.backbone
).to(DEVICE)

# SubCenterArcFace with improved parameters
# k=5 for more intra-class variance, s=30 for stability, m=0.35 for easier separation
metric_head = SubCenterArcFaceHead(EMBEDDING_SIZE, num_classes, k=5, s=30.0, m=0.35).to(DEVICE)

print(f"Model: {args.backbone.upper()} backbone + Sub-Center ArcFace (K=5, s=30, m=0.35)")
print(f"Total parameters: {sum(p.numel() for p in backbone.parameters()):,}")

# --- Optimizer & Loss ---
criterion = nn.CrossEntropyLoss()

# Switch to AdamW optimizer for better convergence
optimizer = optim.AdamW([
    {'params': backbone.parameters()},
    {'params': metric_head.parameters()}
], lr=LR, weight_decay=5e-4)

# CosineAnnealingLR for smooth learning rate decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"Optimizer: AdamW (lr={LR})")
print(f"Scheduler: CosineAnnealingLR")
print(f"{'='*60}\n")

# --- Validation Function ---
def validate(backbone, metric_head, val_loader, criterion, device, return_detailed=False):
    backbone.eval()
    metric_head.eval()
    val_loss = 0

    all_predictions = []
    all_labels = []
    all_top5_correct = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            embeddings = backbone(inputs)
            thetas = metric_head(embeddings, labels)
            loss = criterion(thetas, labels)

            val_loss += loss.item()

            # Top-1 predictions
            _, predicted = torch.max(thetas.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Top-5 predictions
            _, top5_pred = torch.topk(thetas.data, k=min(5, thetas.size(1)), dim=1)
            for i, label in enumerate(labels):
                all_top5_correct.append(label.item() in top5_pred[i].cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * (all_predictions == all_labels).sum() / len(all_labels)
    top5_accuracy = 100. * np.mean(all_top5_correct)

    # F1 scores
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # Precision and Recall
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)

    if return_detailed:
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'predictions': all_predictions,
            'labels': all_labels
        }

    return avg_loss, accuracy, top5_accuracy, f1_macro, f1_weighted

# --- Training Loop ---
os.makedirs(args.output, exist_ok=True)
best_val_f1 = 0.0
best_val_acc = 0.0

# Track metrics history
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'val_top5_acc': [],
    'val_f1_macro': [], 'val_f1_weighted': [],
    'val_precision': [], 'val_recall': []
}

for epoch in range(EPOCHS):
    # Training phase
    backbone.train()
    metric_head.train()
    train_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # 1. Get Embeddings from Backbone
        embeddings = backbone(inputs)

        # 2. Pass Embeddings + Labels to ArcFace Head
        thetas = metric_head(embeddings, labels)

        # 3. Calculate Cross Entropy Loss on the Arced results
        loss = criterion(thetas, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(thetas.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    scheduler.step()

    # Calculate training metrics
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    # Validation phase
    val_loss, val_accuracy, val_top5_acc, val_f1_macro, val_f1_weighted = validate(
        backbone, metric_head, val_loader, criterion, DEVICE
    )

    # Store history
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_accuracy)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_accuracy)
    history['val_top5_acc'].append(val_top5_acc)
    history['val_f1_macro'].append(val_f1_macro)
    history['val_f1_weighted'].append(val_f1_weighted)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_accuracy:.2f}%")
    print(f"  Val:   Loss={val_loss:.4f}, Acc={val_accuracy:.2f}%, Top-5 Acc={val_top5_acc:.2f}%")
    print(f"         F1-Macro={val_f1_macro:.4f}, F1-Weighted={val_f1_weighted:.4f}")

    # Save best model based on F1-macro (better for imbalanced data)
    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        best_val_acc = val_accuracy
        best_model_path = os.path.join(args.output, 'best_model.pth')
        torch.save(backbone.state_dict(), best_model_path)
        print(f"  → Best model saved! (F1-Macro: {val_f1_macro:.4f}, Acc: {val_accuracy:.2f}%)")

    print()

# --- Final Evaluation with Detailed Metrics ---
print(f"\n{'='*60}")
print("FINAL EVALUATION ON VALIDATION SET")
print(f"{'='*60}\n")

# Load best model
backbone.load_state_dict(torch.load(best_model_path))
detailed_results = validate(backbone, metric_head, val_loader, criterion, DEVICE, return_detailed=True)

print(f"Best Model Performance:")
print(f"  Accuracy:       {detailed_results['accuracy']:.2f}%")
print(f"  Top-5 Accuracy: {detailed_results['top5_accuracy']:.2f}%")
print(f"  F1-Macro:       {detailed_results['f1_macro']:.4f}")
print(f"  F1-Weighted:    {detailed_results['f1_weighted']:.4f}")
print(f"  Precision:      {detailed_results['precision']:.4f}")
print(f"  Recall:         {detailed_results['recall']:.4f}")

# Per-class metrics
print(f"\n{'='*60}")
print("PER-CLASS CLASSIFICATION REPORT")
print(f"{'='*60}\n")

class_names = train_dataset.classes
report = classification_report(
    detailed_results['labels'],
    detailed_results['predictions'],
    target_names=class_names,
    zero_division=0
)
print(report)

# Save report
report_path = os.path.join(args.output, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("ARCFACE CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Best Validation F1-Macro: {best_val_f1:.4f}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n\n")
    f.write("="*60 + "\n")
    f.write("PER-CLASS METRICS\n")
    f.write("="*60 + "\n\n")
    f.write(report)

print(f"\nClassification report saved to: {report_path}")

# Confusion Matrix
cm = confusion_matrix(detailed_results['labels'], detailed_results['predictions'])
plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - ArcFace Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
cm_path = os.path.join(args.output, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Confusion matrix saved to: {cm_path}")

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss')
axes[0, 0].plot(history['val_loss'], label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history['train_acc'], label='Train Acc')
axes[0, 1].plot(history['val_acc'], label='Val Acc')
axes[0, 1].plot(history['val_top5_acc'], label='Val Top-5 Acc')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Training and Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# F1 Scores
axes[1, 0].plot(history['val_f1_macro'], label='F1-Macro')
axes[1, 0].plot(history['val_f1_weighted'], label='F1-Weighted')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].set_title('Validation F1 Scores')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Summary
axes[1, 1].axis('off')
summary_text = f"""
FINAL RESULTS

Accuracy:       {detailed_results['accuracy']:.2f}%
Top-5 Accuracy: {detailed_results['top5_accuracy']:.2f}%
F1-Macro:       {detailed_results['f1_macro']:.4f}
F1-Weighted:    {detailed_results['f1_weighted']:.4f}
Precision:      {detailed_results['precision']:.4f}
Recall:         {detailed_results['recall']:.4f}

Classes: {len(class_names)}
Train Samples: {len(train_dataset)}
Val Samples: {len(val_dataset)}
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center')

plt.tight_layout()
history_path = os.path.join(args.output, 'training_history.png')
plt.savefig(history_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Training history saved to: {history_path}")

# --- Save Final Model ---
final_model_path = os.path.join(args.output, 'final_model.pth')
torch.save(backbone.state_dict(), final_model_path)

print(f"\n{'='*60}")
print("TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Best F1-Macro: {best_val_f1:.4f}")
print(f"Best Accuracy: {best_val_acc:.2f}%")
print(f"\nSaved files:")
print(f"  - Best model: {best_model_path}")
print(f"  - Final model: {final_model_path}")
print(f"  - Classification report: {report_path}")
print(f"  - Confusion matrix: {cm_path}")
print(f"  - Training history: {history_path}")
print(f"\nNext steps:")
print(f"  1. Generate prototypes: python ../generate_prototypes.py --model {best_model_path} --backbone {args.backbone}")
print(f"  2. Run cascade inference: python ../cascade_inference.py --image <path>")
print(f"{'='*60}")