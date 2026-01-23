import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_arc import EmbeddingModel, ArcFaceHead, SubCenterArcFaceHead
import os
from tqdm import tqdm
import argparse

# --- Configuration ---
parser = argparse.ArgumentParser(description="Train ArcFace Model")
parser.add_argument('--dataset', type=str, default='../arcface_dataset',
                   help='Path to ArcFace dataset directory')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--embedding-size', type=int, default=512)
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
backbone = EmbeddingModel(embedding_size=EMBEDDING_SIZE, pretrained=True).to(DEVICE)
# SubCenterArcFace with improved parameters
# k=5 for more intra-class variance, s=30 for stability, m=0.35 for easier separation
metric_head = SubCenterArcFaceHead(EMBEDDING_SIZE, num_classes, k=5, s=30.0, m=0.35).to(DEVICE)

print(f"Model: ResNet18 backbone + Sub-Center ArcFace (K=5, s=30, m=0.35)")
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
def validate(backbone, metric_head, val_loader, criterion, device):
    backbone.eval()
    metric_head.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            embeddings = backbone(inputs)
            thetas = metric_head(embeddings, labels)
            loss = criterion(thetas, labels)

            val_loss += loss.item()
            _, predicted = torch.max(thetas.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# --- Training Loop ---
os.makedirs(args.output, exist_ok=True)
best_val_acc = 0.0
best_train_acc = 0.0

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
    val_loss, val_accuracy = validate(backbone, metric_head, val_loader, criterion, DEVICE)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_accuracy:.2f}%")
    print(f"  Val:   Loss={val_loss:.4f}, Acc={val_accuracy:.2f}%")

    # Save best model based on validation accuracy
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_model_path = os.path.join(args.output, 'best_model.pth')
        torch.save(backbone.state_dict(), best_model_path)
        print(f"  → Best model saved! (Val Acc: {val_accuracy:.2f}%)")

    print()

# --- Save Final Model ---
final_model_path = os.path.join(args.output, 'final_model.pth')
torch.save(backbone.state_dict(), final_model_path)

print(f"\n{'='*60}")
print("TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Best model: {best_model_path}")
print(f"Final model: {final_model_path}")
print(f"\nNext steps:")
print(f"  1. Generate prototypes: python ../generate_prototypes.py --model {best_model_path}")
print(f"  2. Run cascade inference: python ../cascade_inference.py --image <path>")
print(f"{'='*60}")