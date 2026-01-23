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
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), # Handle rotation variance
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handle lighting variance
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Data Loading ---
print(f"Loading dataset from: {data_dir}")
dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
num_classes = len(dataset.classes)

print(f"\n{'='*60}")
print(f"ARCFACE TRAINING CONFIGURATION")
print(f"{'='*60}")
print(f"Dataset: {data_dir}")
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LR}")
print(f"Embedding size: {EMBEDDING_SIZE}")
print(f"Device: {DEVICE}")
print(f"{'='*60}\n")

# --- Initialize Models ---
backbone = EmbeddingModel(embedding_size=EMBEDDING_SIZE).to(DEVICE)
# The Head is separate!
# metric_head = ArcFaceHead(EMBEDDING_SIZE, num_classes, s=64.0, m=0.5).to(DEVICE)
# New Line (k=3 allows 3 variations per class)
metric_head = SubCenterArcFaceHead(EMBEDDING_SIZE, num_classes, k=3, s=64.0, m=0.5).to(DEVICE)

# --- Optimizer & Loss ---
criterion = nn.CrossEntropyLoss()
# We optimize BOTH the backbone weights and the Class Centers (metric_head)
optimizer = optim.SGD([
    {'params': backbone.parameters()}, 
    {'params': metric_head.parameters()}
], lr=LR, momentum=0.9, weight_decay=5e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# --- Training Loop ---
os.makedirs(args.output, exist_ok=True)
best_loss = float('inf')

for epoch in range(EPOCHS):
    backbone.train()
    metric_head.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
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

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(thetas.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_path = os.path.join(args.output, 'best_model.pth')
        torch.save(backbone.state_dict(), best_model_path)
        print(f"  → Best model saved: {best_model_path}")

# --- Save Final Model ---
final_model_path = os.path.join(args.output, 'final_model.pth')
torch.save(backbone.state_dict(), final_model_path)

print(f"\n{'='*60}")
print("TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Best model: {best_model_path}")
print(f"Final model: {final_model_path}")
print(f"\nNext steps:")
print(f"  1. Generate prototypes: python ../generate_prototypes.py --model {best_model_path}")
print(f"  2. Run cascade inference: python ../cascade_inference.py --image <path>")