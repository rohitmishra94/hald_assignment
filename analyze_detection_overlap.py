"""
Analyze Multi-Object Detection Issues

Identifies cases where:
1. Under-segmentation: Multiple ground truth objects covered by single detection
2. Over-segmentation: Single ground truth object matched by multiple detections

This script analyzes YOLO detection results to identify these issues without fixing them.
"""

import json
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import os
from pathlib import Path
from collections import defaultdict
import csv
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


def calculate_distance(box1: List[float], box2: List[float]) -> float:
    """Calculate center-to-center distance between two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2
    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2

    return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)


def calculate_coverage(small_box: List[float], large_box: List[float]) -> float:
    """
    Calculate how much of small_box is covered by large_box

    Returns:
        Coverage ratio (0-1)
    """
    x1, y1, w1, h1 = small_box
    x2, y2, w2, h2 = large_box

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
    small_box_area = w1 * h1

    return inter_area / small_box_area if small_box_area > 0 else 0


def analyze_detection_overlap(
    yolo_model_path: str,
    images_dir: str,
    annotation_path: str,
    output_dir: str = 'overlap_analysis_results',
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.5,
    match_iou_threshold: float = 0.5,
    coverage_threshold: float = 0.7  # Consider as covered if >70% overlap
):
    """
    Analyze multi-object detection issues

    Args:
        yolo_model_path: Path to trained YOLO model
        images_dir: Path to images directory
        annotation_path: Path to COCO annotations
        output_dir: Where to save analysis results
        conf_threshold: YOLO confidence threshold
        iou_threshold: YOLO NMS IoU threshold (not used in YOLO26 but kept for compatibility)
        match_iou_threshold: IoU threshold for matching detection to ground truth
        coverage_threshold: Threshold to consider objects as merged
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

    # Storage for issues
    under_segmentation_cases = []
    over_segmentation_cases = []

    print(f"\nAnalyzing {len(image_id_to_info)} images...")
    print("="*80)

    for img_id, img_info in tqdm(image_id_to_info.items(), desc="Analyzing overlap issues"):
        img_path = os.path.join(images_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        # Get ground truth for this image
        gt_annotations = image_id_to_anns[img_id]
        gt_boxes = [ann['bbox'] for ann in gt_annotations]
        gt_ids = [ann['id'] for ann in gt_annotations]

        # Run YOLO detection
        results = model(img_path, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]

        # Extract detected boxes
        detected_boxes = []
        if len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detected_boxes.append([x1, y1, x2 - x1, y2 - y1])

        if len(detected_boxes) == 0 or len(gt_boxes) == 0:
            continue

        # Build IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(detected_boxes)))
        for gt_idx, gt_box in enumerate(gt_boxes):
            for det_idx, det_box in enumerate(detected_boxes):
                iou_matrix[gt_idx, det_idx] = calculate_iou(gt_box, det_box)

        # Identify under-segmentation: Multiple GT boxes matched to single detection
        for det_idx in range(len(detected_boxes)):
            # Find all GT boxes that significantly overlap with this detection
            matched_gts = []
            for gt_idx in range(len(gt_boxes)):
                coverage = calculate_coverage(gt_boxes[gt_idx], detected_boxes[det_idx])
                if coverage >= coverage_threshold:  # GT is largely covered by detection
                    matched_gts.append(gt_idx)

            # If multiple GT boxes are covered by single detection, it's under-segmentation
            if len(matched_gts) >= 2:
                # Calculate proximity between the merged objects
                proximities = []
                for i in range(len(matched_gts)):
                    for j in range(i+1, len(matched_gts)):
                        dist = calculate_distance(gt_boxes[matched_gts[i]], gt_boxes[matched_gts[j]])
                        proximities.append(dist)

                under_segmentation_cases.append({
                    'image_id': img_id,
                    'image_name': img_info['file_name'],
                    'detection_idx': det_idx,
                    'detection_box': detected_boxes[det_idx],
                    'gt_count': len(matched_gts),
                    'gt_indices': matched_gts,
                    'gt_ids': [gt_ids[idx] for idx in matched_gts],
                    'min_distance': min(proximities) if proximities else 0,
                    'avg_distance': np.mean(proximities) if proximities else 0
                })

        # Identify over-segmentation: Single GT matched by multiple detections
        for gt_idx in range(len(gt_boxes)):
            # Find all detections that overlap with this GT
            matched_dets = []
            for det_idx in range(len(detected_boxes)):
                if iou_matrix[gt_idx, det_idx] >= match_iou_threshold:
                    matched_dets.append(det_idx)

            # If single GT is matched by multiple detections, it's over-segmentation
            if len(matched_dets) >= 2:
                # Calculate how spread out the detections are
                det_boxes_for_gt = [detected_boxes[idx] for idx in matched_dets]
                distances = []
                for i in range(len(det_boxes_for_gt)):
                    for j in range(i+1, len(det_boxes_for_gt)):
                        dist = calculate_distance(det_boxes_for_gt[i], det_boxes_for_gt[j])
                        distances.append(dist)

                over_segmentation_cases.append({
                    'image_id': img_id,
                    'image_name': img_info['file_name'],
                    'gt_idx': gt_idx,
                    'gt_id': gt_ids[gt_idx],
                    'gt_box': gt_boxes[gt_idx],
                    'detection_count': len(matched_dets),
                    'detection_indices': matched_dets,
                    'max_distance': max(distances) if distances else 0,
                    'avg_iou': np.mean([iou_matrix[gt_idx, idx] for idx in matched_dets])
                })

    # Generate statistics
    print("\n" + "="*80)
    print("MULTI-OBJECT DETECTION ISSUE ANALYSIS")
    print("="*80)
    print(f"\nTotal images analyzed: {len(image_id_to_info)}")
    print(f"\nUnder-segmentation cases (multiple GT → single detection): {len(under_segmentation_cases)}")
    print(f"Over-segmentation cases (single GT → multiple detections): {len(over_segmentation_cases)}")

    # Save detailed results
    report_path = os.path.join(output_dir, 'overlap_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-OBJECT DETECTION OVERLAP ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {yolo_model_path}\n")
        f.write(f"Confidence Threshold: {conf_threshold}\n")
        f.write(f"Match IoU Threshold: {match_iou_threshold}\n")
        f.write(f"Coverage Threshold: {coverage_threshold}\n\n")

        f.write("="*80 + "\n")
        f.write("UNDER-SEGMENTATION CASES (Multiple Objects → Single Detection)\n")
        f.write("="*80 + "\n")
        f.write(f"Total cases: {len(under_segmentation_cases)}\n\n")

        if under_segmentation_cases:
            # Group by image
            by_image = defaultdict(list)
            for case in under_segmentation_cases:
                by_image[case['image_name']].append(case)

            f.write(f"Images affected: {len(by_image)}\n\n")

            # Statistics
            gt_counts = [case['gt_count'] for case in under_segmentation_cases]
            distances = [case['min_distance'] for case in under_segmentation_cases]

            f.write(f"Average objects merged per detection: {np.mean(gt_counts):.2f}\n")
            f.write(f"Max objects merged into single detection: {max(gt_counts)}\n")
            f.write(f"Average minimum distance between merged objects: {np.mean(distances):.2f} pixels\n\n")

            # Top 10 worst cases
            sorted_cases = sorted(under_segmentation_cases, key=lambda x: x['gt_count'], reverse=True)
            f.write("Top 10 worst under-segmentation cases:\n")
            f.write(f"{'Image':<50} {'Objects Merged':<15} {'Min Distance':<15}\n")
            f.write("-"*80 + "\n")
            for case in sorted_cases[:10]:
                f.write(f"{case['image_name']:<50} {case['gt_count']:<15} {case['min_distance']:<15.2f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("OVER-SEGMENTATION CASES (Single Object → Multiple Detections)\n")
        f.write("="*80 + "\n")
        f.write(f"Total cases: {len(over_segmentation_cases)}\n\n")

        if over_segmentation_cases:
            # Group by image
            by_image = defaultdict(list)
            for case in over_segmentation_cases:
                by_image[case['image_name']].append(case)

            f.write(f"Images affected: {len(by_image)}\n\n")

            # Statistics
            det_counts = [case['detection_count'] for case in over_segmentation_cases]
            distances = [case['max_distance'] for case in over_segmentation_cases]

            f.write(f"Average detections per object: {np.mean(det_counts):.2f}\n")
            f.write(f"Max detections for single object: {max(det_counts)}\n")
            f.write(f"Average spread of detections: {np.mean(distances):.2f} pixels\n\n")

            # Top 10 worst cases
            sorted_cases = sorted(over_segmentation_cases, key=lambda x: x['detection_count'], reverse=True)
            f.write("Top 10 worst over-segmentation cases:\n")
            f.write(f"{'Image':<50} {'Detections':<15} {'Max Distance':<15}\n")
            f.write("-"*80 + "\n")
            for case in sorted_cases[:10]:
                f.write(f"{case['image_name']:<50} {case['detection_count']:<15} {case['max_distance']:<15.2f}\n")

    print(f"\nReport saved to: {report_path}")

    # Save CSV files
    if under_segmentation_cases:
        csv_path = os.path.join(output_dir, 'under_segmentation_cases.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_name', 'detection_idx', 'gt_count', 'min_distance', 'avg_distance'])
            writer.writeheader()
            for case in under_segmentation_cases:
                writer.writerow({
                    'image_name': case['image_name'],
                    'detection_idx': case['detection_idx'],
                    'gt_count': case['gt_count'],
                    'min_distance': case['min_distance'],
                    'avg_distance': case['avg_distance']
                })
        print(f"Under-segmentation cases saved to: {csv_path}")

    if over_segmentation_cases:
        csv_path = os.path.join(output_dir, 'over_segmentation_cases.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_name', 'gt_idx', 'detection_count', 'max_distance', 'avg_iou'])
            writer.writeheader()
            for case in over_segmentation_cases:
                writer.writerow({
                    'image_name': case['image_name'],
                    'gt_idx': case['gt_idx'],
                    'detection_count': case['detection_count'],
                    'max_distance': case['max_distance'],
                    'avg_iou': case['avg_iou']
                })
        print(f"Over-segmentation cases saved to: {csv_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    return {
        'under_segmentation_cases': under_segmentation_cases,
        'over_segmentation_cases': over_segmentation_cases
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Multi-Object Detection Issues")
    parser.add_argument('--yolo-model', type=str,
                       default='yolo_cascade_training/yolo_superclass_plankton/weights/best.pt',
                       help='Path to trained YOLO model')
    parser.add_argument('--images', type=str, default='StudyCase',
                       help='Path to images directory')
    parser.add_argument('--annotations', type=str, default='StudyCase/_annotations.coco.json',
                       help='Path to COCO annotations')
    parser.add_argument('--output', type=str, default='overlap_analysis_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.15,
                       help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='YOLO NMS IoU threshold')
    parser.add_argument('--match-iou', type=float, default=0.5,
                       help='IoU threshold for matching detection to ground truth')
    parser.add_argument('--coverage', type=float, default=0.7,
                       help='Coverage threshold to consider objects as merged')

    args = parser.parse_args()

    analyze_detection_overlap(
        yolo_model_path=args.yolo_model,
        images_dir=args.images,
        annotation_path=args.annotations,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        match_iou_threshold=args.match_iou,
        coverage_threshold=args.coverage
    )
