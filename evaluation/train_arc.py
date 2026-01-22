import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import EmbeddingModel, ArcFaceHead

# --- Configuration ---
data_dir = './dataset/train' # Path to your cropped images
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20
EMBEDDING_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
num_classes = len(dataset.classes)

print(f"Classes: {dataset.classes}")
print(f"Training on {DEVICE} with {num_classes} classes.")

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
for epoch in range(EPOCHS):
    backbone.train()
    metric_head.train()
    total_loss = 0
    
    for inputs, labels in dataloader:
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
        
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

# --- Save Model ---
# Note: We ONLY need the backbone for inference
torch.save(backbone.state_dict(), "arcface_backbone.pth")
print("Training Complete. Backbone saved.")