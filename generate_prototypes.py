import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import os
from tqdm import tqdm
from typing import Dict, List
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# Import ArcFace model
import sys
sys.path.append('evaluation')
from model_arc import EmbeddingModel

def generate_class_prototypes(
    model_path: str,
    dataset_dir: str,
    output_path: str = 'class_prototypes.pth',
    batch_size: int = 32,
    device: str = 'cuda',
    embedding_size: int = 512,
    backbone: str = 'resnet50'
):
    """
    Generate class prototypes by averaging embeddings from training data

    Class prototype = Average of all normalized embeddings for that class

    Args:
        model_path: Path to trained ArcFace model
        dataset_dir: Path to dataset (should have train/ subdirectory)
        output_path: Where to save prototypes
        batch_size: Batch size for inference
        device: Device for inference
        embedding_size: Embedding dimension
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Generating prototypes on {device}...")

    # Load model
    print(f"Loading ArcFace model ({backbone})...")
    model = EmbeddingModel(embedding_size=embedding_size, pretrained=False, backbone=backbone)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(dataset_dir, 'train')
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")

    # Extract embeddings
    print("Extracting embeddings...")
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)

            # Get embeddings
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Extracted {len(all_embeddings)} embeddings")

    # Calculate class prototypes (average per class)
    print("Calculating class prototypes...")
    num_classes = len(dataset.classes)
    prototypes = torch.zeros(num_classes, embedding_size)

    for class_idx in range(num_classes):
        class_mask = (all_labels == class_idx)
        class_embeddings = all_embeddings[class_mask]

        if len(class_embeddings) > 0:
            # Average and normalize
            prototype = class_embeddings.mean(dim=0)
            prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1).squeeze(0)
            prototypes[class_idx] = prototype

            print(f"  {dataset.classes[class_idx]}: {len(class_embeddings)} samples")

    # Save prototypes
    prototype_dict = {
        'embeddings': prototypes.to(device),
        'labels': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'idx_to_class': {v: k for k, v in dataset.class_to_idx.items()},
        'embedding_size': embedding_size,
        'num_classes': num_classes
    }

    torch.save(prototype_dict, output_path)
    print(f"\nPrototypes saved to: {output_path}")

    # Calculate statistics
    calculate_prototype_statistics(
        prototypes,
        dataset.classes,
        output_dir=os.path.dirname(output_path)
    )

    # Visualize embeddings
    visualize_embeddings(
        all_embeddings.numpy(),
        all_labels.numpy(),
        prototypes.numpy(),
        dataset.classes,
        output_dir=os.path.dirname(output_path)
    )

    return prototype_dict

def calculate_prototype_statistics(
    prototypes: torch.Tensor,
    class_names: List[str],
    output_dir: str = '.'
):
    """
    Calculate and save statistics about class prototypes
    """
    print("\nCalculating prototype statistics...")

    # Calculate pairwise cosine similarities
    similarities = F.cosine_similarity(
        prototypes.unsqueeze(1),
        prototypes.unsqueeze(0),
        dim=2
    )

    # Find most similar class pairs (excluding diagonal)
    similarities_np = similarities.numpy()
    np.fill_diagonal(similarities_np, -1)

    stats = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            stats.append({
                'class1': class_names[i],
                'class2': class_names[j],
                'similarity': similarities_np[i, j]
            })

    # Sort by similarity (most similar first)
    stats.sort(key=lambda x: x['similarity'], reverse=True)

    # Save statistics
    stats_path = os.path.join(output_dir, 'prototype_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASS PROTOTYPE STATISTICS\n")
        f.write("="*60 + "\n\n")

        f.write("Most Similar Class Pairs (Top 20):\n")
        f.write("-"*60 + "\n")
        for stat in stats[:20]:
            f.write(f"{stat['class1']:<25} <-> {stat['class2']:<25} {stat['similarity']:.4f}\n")

        f.write("\n")
        f.write("Least Similar Class Pairs (Bottom 20):\n")
        f.write("-"*60 + "\n")
        for stat in stats[-20:]:
            f.write(f"{stat['class1']:<25} <-> {stat['class2']:<25} {stat['similarity']:.4f}\n")

        f.write("\n")
        f.write("="*60 + "\n")
        f.write("NOTES:\n")
        f.write("="*60 + "\n")
        f.write("- High similarity (>0.9): Classes may be confused\n")
        f.write("- Consider higher angular margin for similar classes\n")
        f.write("- Consider data augmentation for better separation\n")

    print(f"Statistics saved to: {stats_path}")

    # Print summary
    print(f"\nMost similar classes: {stats[0]['class1']} <-> {stats[0]['class2']} ({stats[0]['similarity']:.4f})")
    print(f"Least similar classes: {stats[-1]['class1']} <-> {stats[-1]['class2']} ({stats[-1]['similarity']:.4f})")
    print(f"Average inter-class similarity: {np.mean([s['similarity'] for s in stats]):.4f}")

def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    class_names: List[str],
    output_dir: str = '.',
    n_samples_per_class: int = 100
):
    """
    Visualize embeddings using UMAP and t-SNE
    """
    print("\nVisualizing embeddings...")

    # Sample embeddings for visualization (to speed up)
    sampled_indices = []
    for class_idx in range(len(class_names)):
        class_mask = (labels == class_idx)
        class_indices = np.where(class_mask)[0]

        if len(class_indices) > n_samples_per_class:
            sampled = np.random.choice(class_indices, n_samples_per_class, replace=False)
        else:
            sampled = class_indices

        sampled_indices.extend(sampled)

    sampled_embeddings = embeddings[sampled_indices]
    sampled_labels = labels[sampled_indices]

    # Combine with prototypes
    combined_embeddings = np.vstack([sampled_embeddings, prototypes])
    combined_labels = np.concatenate([
        sampled_labels,
        np.arange(len(prototypes))
    ])

    # UMAP projection
    print("  Running UMAP...")
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(combined_embeddings)

    # Split back into samples and prototypes
    n_samples = len(sampled_embeddings)
    umap_samples = umap_embeddings[:n_samples]
    umap_prototypes = umap_embeddings[n_samples:]

    # Plot UMAP
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot samples
    for class_idx in range(len(class_names)):
        mask = (sampled_labels == class_idx)
        ax.scatter(
            umap_samples[mask, 0],
            umap_samples[mask, 1],
            alpha=0.3,
            s=20,
            label=class_names[class_idx]
        )

    # Plot prototypes
    ax.scatter(
        umap_prototypes[:, 0],
        umap_prototypes[:, 1],
        c='red',
        marker='*',
        s=300,
        edgecolors='black',
        linewidths=2,
        label='Prototypes',
        zorder=10
    )

    # Add labels to prototypes
    for idx, (x, y) in enumerate(umap_prototypes):
        ax.annotate(
            class_names[idx],
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
        )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('ArcFace Embedding Space (UMAP)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    umap_path = os.path.join(output_dir, 'embedding_visualization_umap.png')
    plt.savefig(umap_path, dpi=150, bbox_inches='tight')
    print(f"  UMAP visualization saved to: {umap_path}")
    plt.close()

    print("Visualization complete!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Class Prototypes")
    parser.add_argument('--model', type=str, default='arcface_models/best_model.pth',
                       help='Path to trained ArcFace model')
    parser.add_argument('--dataset', type=str, default='arcface_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='arcface_models/class_prototypes.pth',
                       help='Output path for prototypes')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50'],
                       help='Backbone architecture (must match training)')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Generate prototypes
    prototypes = generate_class_prototypes(
        model_path=args.model,
        dataset_dir=args.dataset,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        backbone=args.backbone
    )

    print("\n" + "="*60)
    print("PROTOTYPE GENERATION COMPLETE!")
    print("="*60)
    print(f"\nPrototypes saved to: {args.output}")
    print(f"Number of classes: {prototypes['num_classes']}")
    print(f"Embedding size: {prototypes['embedding_size']}")
    print("\nReady for cascade inference!")
    print(f"Run: python cascade_inference.py --image <path>")
