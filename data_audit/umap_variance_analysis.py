"""
Enhanced UMAP Visualization with PCA Variance Analysis
Add this to your intra_class_analysis.ipynb notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import umap


def visualize_with_umap_and_pca_variance(
    images: List[np.ndarray],
    class_name: str,
    device: str = 'cuda',
    show_variance_info: bool = True,
    show_ellipse: bool = True,
    n_pca_components: int = 50
):
    """
    Enhanced UMAP visualization with PCA variance analysis

    Returns eigenvalues, explained variance, and intra-class variance metrics
    """
    # Check device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Analyzing {len(images)} samples for {class_name}...")

    # Extract features using ResNet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Batch processing
    batch_size = 32
    features = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_tensors = torch.stack([transform(img) for img in batch_images])
            batch_tensors = batch_tensors.to(device)

            batch_features = feature_extractor(batch_tensors)
            batch_features = batch_features.squeeze().cpu().numpy()

            if len(batch_features.shape) == 1:
                batch_features = batch_features.reshape(1, -1)

            features.append(batch_features)

    features = np.vstack(features)
    print(f"Feature shape: {features.shape}")

    # ===== PCA ANALYSIS =====
    print("\nPerforming PCA analysis...")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Full PCA for variance analysis
    pca_full = PCA()
    pca_features_full = pca_full.fit_transform(features_scaled)

    # Get variance metrics
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    eigenvalues = pca_full.explained_variance_

    # Calculate total variance in feature space
    total_variance = np.sum(np.var(features, axis=0))
    std_dev = np.std(features, axis=0)
    mean_std = np.mean(std_dev)

    # Variance in first N components
    n_comp = min(n_pca_components, len(eigenvalues))
    variance_first_n = np.sum(explained_variance_ratio[:n_comp])

    print(f"\nVariance Analysis:")
    print(f"  Total variance (feature space): {total_variance:.2f}")
    print(f"  Mean std deviation: {mean_std:.4f}")
    print(f"  Variance explained by PC1: {explained_variance_ratio[0]:.4f} ({explained_variance_ratio[0]*100:.2f}%)")
    print(f"  Variance explained by PC1-2: {cumulative_variance[1]:.4f} ({cumulative_variance[1]*100:.2f}%)")
    print(f"  Variance explained by PC1-{n_comp}: {variance_first_n:.4f} ({variance_first_n*100:.2f}%)")
    print(f"  Top 3 eigenvalues: {eigenvalues[:3]}")

    # ===== UMAP EMBEDDING =====
    print("\nRunning UMAP...")
    try:
        from cuml.manifold import UMAP as cumlUMAP
        reducer = cumlUMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        print("Using GPU-accelerated UMAP")
    except ImportError:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        print("Using CPU UMAP")

    embedding = reducer.fit_transform(features_scaled)

    # ===== PCA ON UMAP 2D EMBEDDINGS =====
    print("\nPerforming PCA on UMAP 2D embeddings...")

    # PCA on the 2D UMAP space
    pca_umap = PCA(n_components=2)
    pca_umap.fit(embedding)

    # Get UMAP variance metrics
    umap_eigenvalues = pca_umap.explained_variance_
    umap_variance_ratio = pca_umap.explained_variance_ratio_
    umap_eigenvectors = pca_umap.components_

    # Aspect ratio (elongation measure)
    aspect_ratio = umap_eigenvalues[0] / umap_eigenvalues[1] if umap_eigenvalues[1] > 0 else np.inf
    directionality = umap_variance_ratio[0]  # How directional is the spread

    # Other UMAP metrics
    centroid = np.mean(embedding, axis=0)
    distances_from_centroid = np.linalg.norm(embedding - centroid, axis=1)
    mean_distance = np.mean(distances_from_centroid)
    spread_umap = np.std(embedding)

    print(f"\nUMAP 2D PCA Analysis:")
    print(f"  UMAP PC1 variance: {umap_variance_ratio[0]:.4f} ({umap_variance_ratio[0]*100:.2f}%)")
    print(f"  UMAP PC2 variance: {umap_variance_ratio[1]:.4f} ({umap_variance_ratio[1]*100:.2f}%)")
    print(f"  UMAP eigenvalues: [{umap_eigenvalues[0]:.4f}, {umap_eigenvalues[1]:.4f}]")
    print(f"  Aspect ratio (λ1/λ2): {aspect_ratio:.2f}")
    print(f"  Directionality: {directionality:.4f} (1.0=linear, 0.5=circular)")

    print(f"\nUMAP Space Metrics:")
    print(f"  Mean distance from centroid: {mean_distance:.4f}")
    print(f"  Spread (std): {spread_umap:.4f}")

    # ===== VISUALIZATION =====
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: UMAP with color by distance from centroid
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=distances_from_centroid,
        cmap='viridis',
        alpha=0.6,
        s=30
    )
    ax1.plot(centroid[0], centroid[1], 'r*', markersize=20, label='Centroid')

    if show_ellipse and len(embedding) > 2:
        # 95% confidence ellipse
        cov = np.cov(embedding.T)
        eigenvalues_ellipse, eigenvectors_ellipse = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors_ellipse[1, 0], eigenvectors_ellipse[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues_ellipse) * 2.447  # 95% confidence

        ellipse = Ellipse(
            centroid, width, height,
            angle=angle,
            facecolor='none',
            edgecolor='red',
            linewidth=2,
            label='95% CI'
        )
        ax1.add_patch(ellipse)

        # Draw PCA axes (principal directions)
        scale_factor = 3  # Scale for visibility
        for i, (eigvec, eigval) in enumerate(zip(umap_eigenvectors, umap_eigenvalues)):
            # Draw arrow from centroid along principal component
            ax1.arrow(
                centroid[0], centroid[1],
                eigvec[0] * np.sqrt(eigval) * scale_factor,
                eigvec[1] * np.sqrt(eigval) * scale_factor,
                head_width=0.15, head_length=0.2,
                fc='orange' if i == 0 else 'cyan',
                ec='orange' if i == 0 else 'cyan',
                linewidth=2.5,
                alpha=0.8,
                label=f'PC{i+1} ({umap_variance_ratio[i]*100:.1f}%)'
            )

    ax1.set_xlabel('UMAP 1', fontsize=12)
    ax1.set_ylabel('UMAP 2', fontsize=12)
    ax1.set_title(f'{class_name} - UMAP with PCA axes (aspect ratio: {aspect_ratio:.2f})', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Distance from centroid')

    # Plot 2: Scree plot (variance explained)
    ax2 = fig.add_subplot(gs[0, 1])
    n_components_plot = min(50, len(explained_variance_ratio))
    ax2.plot(range(1, n_components_plot + 1),
             explained_variance_ratio[:n_components_plot],
             'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Variance Explained', fontsize=12)
    ax2.set_title(f'Scree Plot - Individual Variance', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
    ax2.legend()

    # Plot 3: Cumulative variance explained
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(range(1, n_components_plot + 1),
             cumulative_variance[:n_components_plot],
             'ro-', linewidth=2, markersize=8)
    ax3.axhline(y=0.9, color='g', linestyle='--', label='90% variance')
    ax3.axhline(y=0.95, color='b', linestyle='--', label='95% variance')
    ax3.set_xlabel('Number of Components', fontsize=12)
    ax3.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax3.set_title(f'Cumulative Variance Explained', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: PC1 vs PC2
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(pca_features_full[:, 0], pca_features_full[:, 1],
                c=distances_from_centroid, cmap='viridis', alpha=0.6, s=30)
    ax4.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)', fontsize=12)
    ax4.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)', fontsize=12)
    ax4.set_title(f'First 2 Principal Components', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot 5: Eigenvalue bar chart (top 20)
    ax5 = fig.add_subplot(gs[1, 1])
    n_bars = min(20, len(eigenvalues))
    ax5.bar(range(1, n_bars + 1), eigenvalues[:n_bars], color='steelblue', alpha=0.7)
    ax5.set_xlabel('Principal Component', fontsize=12)
    ax5.set_ylabel('Eigenvalue', fontsize=12)
    ax5.set_title(f'Top {n_bars} Eigenvalues', fontsize=14)
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Variance statistics summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_text = f"""
    VARIANCE ANALYSIS SUMMARY
    {'-' * 40}

    Class: {class_name}
    Samples: {len(images)}

    FEATURE SPACE (2048D)
    Total Variance: {total_variance:.2f}
    Mean Std Dev: {mean_std:.4f}
    PC1-50: {variance_first_n*100:.2f}%

    UMAP 2D PCA
    PC1 variance: {umap_variance_ratio[0]*100:.2f}%
    PC2 variance: {umap_variance_ratio[1]*100:.2f}%
    Eigenvalues: [{umap_eigenvalues[0]:.2f}, {umap_eigenvalues[1]:.2f}]
    Aspect ratio: {aspect_ratio:.2f}
    Directionality: {directionality:.3f}

    UMAP SPREAD
    Mean dist from center: {mean_distance:.4f}
    Spread (std): {spread_umap:.4f}

    INTERPRETATION
    Aspect ratio > 2: Elongated (directional)
    Aspect ratio ≈ 1: Circular (isotropic)
    High eigenvalues: High variance class
    Low eigenvalues: Compact class
    Directionality > 0.7: Strong single direction
    """

    ax6.text(0.1, 0.95, summary_text,
             transform=ax6.transAxes,
             fontsize=11,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Intra-Class Variance Analysis: {class_name}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.show()

    return {
        'embedding': embedding,
        'features': features,
        'pca_features': pca_features_full,
        'eigenvalues': eigenvalues,
        'eigenvectors': pca_full.components_,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'total_variance': total_variance,
        'mean_std': mean_std,
        'centroid': centroid,
        'spread_umap': spread_umap,
        'mean_distance_from_centroid': mean_distance,
        # UMAP 2D PCA metrics
        'umap_eigenvalues': umap_eigenvalues,
        'umap_variance_ratio': umap_variance_ratio,
        'umap_eigenvectors': umap_eigenvectors,
        'aspect_ratio': aspect_ratio,
        'directionality': directionality
    }


def compare_classes_variance(
    class_names: List[str],
    coco_data: Dict,
    images_dir: str,
    max_samples_per_class: int = 100,
    device: str = 'cuda'
):
    """
    Compare intra-class variance across multiple classes
    Identifies which classes have high variance (challenging) vs low variance (cohesive)
    """
    from typing import Dict as typing_dict
    import json

    print("="*60)
    print("COMPARATIVE VARIANCE ANALYSIS ACROSS CLASSES")
    print("="*60)

    results = {}

    # Extract objects function (assuming it exists)
    def extract_objects_by_class(class_name, coco_data, images_dir, max_objects):
        category_id = next((cat['id'] for cat in coco_data['categories']
                           if cat['name'] == class_name), None)
        class_annotations = [ann for ann in coco_data['annotations']
                           if ann['category_id'] == category_id][:max_objects]
        image_map = {img['id']: img for img in coco_data['images']}

        import cv2
        extracted_objects = []
        for ann in class_annotations:
            bbox = ann['bbox']
            image_info = image_map[ann['image_id']]
            img_path = f"{images_dir}/{image_info['file_name']}"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cropped = img[y:y+h, x:x+w]
            extracted_objects.append(cropped)

        return extracted_objects

    for class_name in class_names:
        print(f"\nAnalyzing {class_name}...")
        objects = extract_objects_by_class(class_name, coco_data, images_dir, max_samples_per_class)

        if len(objects) < 5:
            print(f"  Skipping (only {len(objects)} samples)")
            continue

        result = visualize_with_umap_and_pca_variance(
            objects, class_name, device=device, show_variance_info=False
        )

        results[class_name] = {
            'n_samples': len(objects),
            'total_variance': result['total_variance'],
            'mean_std': result['mean_std'],
            'pc1_variance': result['explained_variance_ratio'][0],
            'pc1_2_variance': result['cumulative_variance'][1],
            'eigenvalue_1': result['eigenvalues'][0],
            'umap_spread': result['spread_umap'],
            'mean_distance': result['mean_distance_from_centroid'],
            # UMAP 2D PCA metrics
            'umap_pc1': result['umap_variance_ratio'][0],
            'umap_eigenval1': result['umap_eigenvalues'][0],
            'aspect_ratio': result['aspect_ratio'],
            'directionality': result['directionality']
        }

    # Create comparison table
    print("\n" + "="*120)
    print("VARIANCE COMPARISON TABLE")
    print("="*120)
    print(f"{'Class':<25} {'Samples':<10} {'Total Var':<12} {'UMAP λ1':<12} {'Aspect':<10} {'Direct':<10} {'UMAP PC1%':<12}")
    print("-"*120)

    sorted_classes = sorted(results.items(), key=lambda x: x[1]['aspect_ratio'], reverse=True)

    for class_name, metrics in sorted_classes:
        print(f"{class_name:<25} {metrics['n_samples']:<10} "
              f"{metrics['total_variance']:<12.2f} {metrics['umap_eigenval1']:<12.2f} "
              f"{metrics['aspect_ratio']:<10.2f} {metrics['directionality']:<10.3f} "
              f"{metrics['umap_pc1']*100:<12.2f}")

    print("="*120)

    # Identify high and low variance classes
    variances = [m['total_variance'] for m in results.values()]
    mean_variance = np.mean(variances)
    std_variance = np.std(variances)

    high_variance = [c for c, m in results.items() if m['total_variance'] > mean_variance + std_variance]
    low_variance = [c for c, m in results.items() if m['total_variance'] < mean_variance - std_variance]

    print(f"\nHIGH VARIANCE CLASSES (challenging, need more K in Sub-Center ArcFace):")
    for c in high_variance:
        print(f"  • {c}: {results[c]['total_variance']:.2f}")

    print(f"\nLOW VARIANCE CLASSES (cohesive, easy to learn):")
    for c in low_variance:
        print(f"  • {c}: {results[c]['total_variance']:.2f}")

    print(f"\nRECOMMENDATIONS:")
    print(f"  • High variance classes may benefit from K=10 sub-centers")
    print(f"  • Consider more training samples for high variance classes")
    print(f"  • Low variance classes are well-represented")

    return results


# Example usage:
"""
# For single class analysis
objects = extract_objects_by_class('Chlorella sp', coco_data, images_dir, max_objects=200)
result = visualize_with_umap_and_pca_variance(objects, 'Chlorella sp')

# For comparative analysis
class_names = ['Chlorella sp', 'Oscillatoria sp', 'Prymnesium sp', 'Pyramimonas sp']
comparison = compare_classes_variance(class_names, coco_data, images_dir, max_samples_per_class=100)
"""
