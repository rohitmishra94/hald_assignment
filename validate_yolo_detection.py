"""
Validate YOLO Detection Stage (Stage 1)

Compares YOLO detections against ground truth annotations to check:
1. Detection recall (are we finding all objects?)
2. False positives (ghost detections)
3. Per-image object count accuracy
4. IoU distribution

This validates Stage 1 of the cascade before ArcFace classification.
"""

import json
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_coco_annotations(annotation_path: str) -> Dict:
    """Load COCO format annotations"""
    with open(annotation_path, 'r') as f:
        return json.load(f)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two boxes

    Args:
        box1, box2: [x, y, width, height] in COCO format

    Returns:
        IoU score (0-1)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2]
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2

    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def validate_yolo_detection(
    yolo_model_path: str,
    images_dir: str,
    annotation_path: str,
    output_dir: str = 'yolo_validation_results',
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.5,
    match_iou_threshold: float = 0.5
):
    """
    Validate YOLO detection against ground truth

    Args:
        yolo_model_path: Path to trained YOLO model
        images_dir: Path to images directory
        annotation_path: Path to COCO annotations
        output_dir: Where to save validation results
        conf_threshold: YOLO confidence threshold
        iou_threshold: YOLO NMS IoU threshold
        match_iou_threshold: IoU threshold for matching detection to ground truth
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model
    print(f"Loading YOLO model from {yolo_model_path}...")
    model = YOLO(yolo_model_path)

    # Load annotations
    print(f"Loading annotations from {annotation_path}...")
    coco_data = load_coco_annotations(annotation_path)

    # Create image ID to annotations mapping
    image_id_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id_to_anns[ann['image_id']].append(ann)

    # Create image ID to image info mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Statistics
    total_gt_objects = 0
    total_detected_objects = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    per_image_results = []
    iou_scores = []

    print(f"\nProcessing {len(image_id_to_info)} images...")
    print("="*80)

    for img_id, img_info in tqdm(image_id_to_info.items(), desc="Validating detections"):
        img_path = os.path.join(images_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        # Get ground truth for this image
        gt_annotations = image_id_to_anns[img_id]
        gt_boxes = [ann['bbox'] for ann in gt_annotations]
        num_gt = len(gt_boxes)

        # Run YOLO detection
        results = model(img_path, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]

        # Extract detected boxes (convert from xyxy to xywh)
        detected_boxes = []
        if len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detected_boxes.append([x1, y1, x2 - x1, y2 - y1])

        num_detected = len(detected_boxes)

        # Match detections to ground truth
        matched_gt = set()
        matched_det = set()

        for det_idx, det_box in enumerate(detected_boxes):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= match_iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_det.add(det_idx)
                iou_scores.append(best_iou)

        # Calculate metrics for this image
        true_positives = len(matched_gt)
        false_positives = num_detected - len(matched_det)
        false_negatives = num_gt - len(matched_gt)

        # Update totals
        total_gt_objects += num_gt
        total_detected_objects += num_detected
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        # Store per-image result
        per_image_results.append({
            'image_id': img_id,
            'image_name': img_info['file_name'],
            'gt_count': num_gt,
            'detected_count': num_detected,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'recall': true_positives / num_gt if num_gt > 0 else 0,
            'precision': true_positives / num_detected if num_detected > 0 else 0
        })

    # Calculate overall metrics
    overall_precision = total_true_positives / total_detected_objects if total_detected_objects > 0 else 0
    overall_recall = total_true_positives / total_gt_objects if total_gt_objects > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    # Print summary
    print("\n" + "="*80)
    print("YOLO DETECTION VALIDATION SUMMARY")
    print("="*80)
    print(f"\nGround Truth Objects:     {total_gt_objects}")
    print(f"Detected Objects:         {total_detected_objects}")
    print(f"True Positives (Matched): {total_true_positives}")
    print(f"False Positives (Ghosts): {total_false_positives}")
    print(f"False Negatives (Missed): {total_false_negatives}")
    print(f"\nOverall Precision:        {overall_precision:.4f} ({overall_precision*100:.2f}%)")
    print(f"Overall Recall:           {overall_recall:.4f} ({overall_recall*100:.2f}%)")
    print(f"Overall F1-Score:         {overall_f1:.4f}")
    print(f"\nMean IoU (matched boxes): {np.mean(iou_scores):.4f}" if iou_scores else "N/A")
    print("="*80)

    # Save detailed report
    report_path = os.path.join(output_dir, 'detection_validation_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("YOLO DETECTION STAGE VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {yolo_model_path}\n")
        f.write(f"Confidence Threshold: {conf_threshold}\n")
        f.write(f"IoU Threshold (NMS): {iou_threshold}\n")
        f.write(f"Match IoU Threshold: {match_iou_threshold}\n\n")

        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Ground Truth Objects:     {total_gt_objects}\n")
        f.write(f"Detected Objects:         {total_detected_objects}\n")
        f.write(f"True Positives (Matched): {total_true_positives}\n")
        f.write(f"False Positives (Ghosts): {total_false_positives}\n")
        f.write(f"False Negatives (Missed): {total_false_negatives}\n\n")

        f.write(f"Overall Precision:        {overall_precision:.4f} ({overall_precision*100:.2f}%)\n")
        f.write(f"Overall Recall:           {overall_recall:.4f} ({overall_recall*100:.2f}%)\n")
        f.write(f"Overall F1-Score:         {overall_f1:.4f}\n")
        f.write(f"Mean IoU (matched):       {np.mean(iou_scores):.4f}\n\n" if iou_scores else "N/A\n\n")

        f.write("="*80 + "\n")
        f.write("PER-IMAGE RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"{'Image':<40} {'GT':>5} {'Det':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Recall':>8} {'Prec':>8}\n")
        f.write("-"*80 + "\n")

        for result in per_image_results:
            f.write(f"{result['image_name']:<40} "
                   f"{result['gt_count']:>5} "
                   f"{result['detected_count']:>5} "
                   f"{result['true_positives']:>5} "
                   f"{result['false_positives']:>5} "
                   f"{result['false_negatives']:>5} "
                   f"{result['recall']:>8.4f} "
                   f"{result['precision']:>8.4f}\n")

        # Find problematic images
        f.write("\n" + "="*80 + "\n")
        f.write("IMAGES WITH HIGH FALSE POSITIVES (Ghosts)\n")
        f.write("="*80 + "\n")
        high_fp = sorted([r for r in per_image_results if r['false_positives'] >= 5],
                        key=lambda x: x['false_positives'], reverse=True)
        if high_fp:
            for result in high_fp[:10]:
                f.write(f"{result['image_name']:<40} FP={result['false_positives']:>3} (detected {result['detected_count']}, actual {result['gt_count']})\n")
        else:
            f.write("No images with significant false positives.\n")

        f.write("\n" + "="*80 + "\n")
        f.write("IMAGES WITH HIGH FALSE NEGATIVES (Missing Objects)\n")
        f.write("="*80 + "\n")
        high_fn = sorted([r for r in per_image_results if r['false_negatives'] >= 5],
                        key=lambda x: x['false_negatives'], reverse=True)
        if high_fn:
            for result in high_fn[:10]:
                f.write(f"{result['image_name']:<40} FN={result['false_negatives']:>3} (detected {result['detected_count']}, actual {result['gt_count']})\n")
        else:
            f.write("No images with significant missed objects.\n")

    print(f"\nDetailed report saved to: {report_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Detection count distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: GT vs Detected counts
    ax = axes[0, 0]
    gt_counts = [r['gt_count'] for r in per_image_results]
    det_counts = [r['detected_count'] for r in per_image_results]
    ax.scatter(gt_counts, det_counts, alpha=0.5)
    ax.plot([0, max(gt_counts)], [0, max(gt_counts)], 'r--', label='Perfect detection')
    ax.set_xlabel('Ground Truth Count')
    ax.set_ylabel('Detected Count')
    ax.set_title('Detection Count: Ground Truth vs Detected')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Recall distribution
    ax = axes[0, 1]
    recalls = [r['recall'] for r in per_image_results]
    ax.hist(recalls, bins=20, edgecolor='black')
    ax.axvline(np.mean(recalls), color='r', linestyle='--', label=f'Mean: {np.mean(recalls):.3f}')
    ax.set_xlabel('Recall (per image)')
    ax.set_ylabel('Frequency')
    ax.set_title('Per-Image Recall Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: False positives distribution
    ax = axes[1, 0]
    fps = [r['false_positives'] for r in per_image_results]
    ax.hist(fps, bins=max(fps)+1 if fps else 1, edgecolor='black')
    ax.axvline(np.mean(fps), color='r', linestyle='--', label=f'Mean: {np.mean(fps):.2f}')
    ax.set_xlabel('False Positives (Ghosts)')
    ax.set_ylabel('Frequency')
    ax.set_title('False Positives per Image')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: False negatives distribution
    ax = axes[1, 1]
    fns = [r['false_negatives'] for r in per_image_results]
    ax.hist(fns, bins=max(fns)+1 if fns else 1, edgecolor='black')
    ax.axvline(np.mean(fns), color='r', linestyle='--', label=f'Mean: {np.mean(fns):.2f}')
    ax.set_xlabel('False Negatives (Missed)')
    ax.set_ylabel('Frequency')
    ax.set_title('False Negatives per Image')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'detection_validation_plots.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Visualizations saved to: {viz_path}")
    plt.close()

    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")

    return {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'total_gt': total_gt_objects,
        'total_detected': total_detected_objects,
        'true_positives': total_true_positives,
        'false_positives': total_false_positives,
        'false_negatives': total_false_negatives,
        'per_image_results': per_image_results
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate YOLO Detection Stage")
    parser.add_argument('--yolo-model', type=str,
                       default='yolo_cascade_training/yolo_superclass_plankton/weights/best.pt',
                       help='Path to trained YOLO model')
    parser.add_argument('--images', type=str, default='StudyCase',
                       help='Path to images directory')
    parser.add_argument('--annotations', type=str, default='StudyCase/_annotations.coco.json',
                       help='Path to COCO annotations')
    parser.add_argument('--output', type=str, default='yolo_validation_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.15,
                       help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='YOLO NMS IoU threshold')
    parser.add_argument('--match-iou', type=float, default=0.5,
                       help='IoU threshold for matching detection to ground truth')

    args = parser.parse_args()

    validate_yolo_detection(
        yolo_model_path=args.yolo_model,
        images_dir=args.images,
        annotation_path=args.annotations,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        match_iou_threshold=args.match_iou
    )
