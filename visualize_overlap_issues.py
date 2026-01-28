"""
Visualize Multi-Object Detection Issues

Visualizes cases where:
1. Under-segmentation: Multiple ground truth objects covered by single detection (Orange)
2. Over-segmentation: Single ground truth object matched by multiple detections (Blue)

Reads analysis results from analyze_detection_overlap.py and creates annotated images.
"""

import json
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_coco_annotations(annotation_path: str) -> Dict:
    """Load COCO format annotations"""
    with open(annotation_path, 'r') as f:
        return json.load(f)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_coverage(small_box: List[float], large_box: List[float]) -> float:
    """Calculate how much of small_box is covered by large_box"""
    x1, y1, w1, h1 = small_box
    x2, y2, w2, h2 = large_box

    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    small_box_area = w1 * h1

    return inter_area / small_box_area if small_box_area > 0 else 0


def draw_box(img, box, color, thickness=2, label=None):
    """Draw a bounding box on image"""
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    if label:
        # Add label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y), color, -1)
        cv2.putText(img, label, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)


def visualize_overlap_issues(
    yolo_model_path: str,
    images_dir: str,
    annotation_path: str,
    output_dir: str = 'overlap_analysis_results/visualized_issues',
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.5,
    match_iou_threshold: float = 0.5,
    coverage_threshold: float = 0.7,
    max_images_per_type: int = 20
):
    """
    Visualize multi-object detection issues

    Colors:
    - Green: Ground truth boxes (normal)
    - Red: Detection boxes (normal)
    - Orange (thick): Under-segmentation (detection covering multiple GT)
    - Blue (thick): Over-segmentation (GT matched by multiple detections)
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model
    print(f"Loading YOLO model from {yolo_model_path}...")
    model = YOLO(yolo_model_path)

    # Load annotations
    print(f"Loading annotations from {annotation_path}...")
    coco_data = load_coco_annotations(annotation_path)

    # Create mappings
    image_id_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id_to_anns[ann['image_id']].append(ann)

    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Track issues
    under_seg_images = set()
    over_seg_images = set()
    under_seg_count = 0
    over_seg_count = 0

    print(f"\nAnalyzing and visualizing {len(image_id_to_info)} images...")
    print("="*80)

    for img_id, img_info in tqdm(image_id_to_info.items(), desc="Processing images"):
        img_path = os.path.join(images_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        # Get ground truth
        gt_annotations = image_id_to_anns[img_id]
        gt_boxes = [ann['bbox'] for ann in gt_annotations]

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

        # Find under-segmentation cases
        under_seg_detections = set()
        under_seg_gt_groups = {}

        for det_idx in range(len(detected_boxes)):
            matched_gts = []
            for gt_idx in range(len(gt_boxes)):
                coverage = calculate_coverage(gt_boxes[gt_idx], detected_boxes[det_idx])
                if coverage >= coverage_threshold:
                    matched_gts.append(gt_idx)

            if len(matched_gts) >= 2:
                under_seg_detections.add(det_idx)
                under_seg_gt_groups[det_idx] = matched_gts

        # Find over-segmentation cases
        over_seg_gts = set()
        over_seg_det_groups = {}

        for gt_idx in range(len(gt_boxes)):
            matched_dets = []
            for det_idx in range(len(detected_boxes)):
                if iou_matrix[gt_idx, det_idx] >= match_iou_threshold:
                    matched_dets.append(det_idx)

            if len(matched_dets) >= 2:
                over_seg_gts.add(gt_idx)
                over_seg_det_groups[gt_idx] = matched_dets

        # If there are any issues, visualize this image
        has_issues = len(under_seg_detections) > 0 or len(over_seg_gts) > 0

        if has_issues:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Draw all ground truth boxes in green (thin)
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in over_seg_gts:
                    continue  # Will draw these in blue later
                draw_box(img, gt_box, (0, 255, 0), thickness=1)

            # Draw all detection boxes in red (thin)
            for det_idx, det_box in enumerate(detected_boxes):
                if det_idx in under_seg_detections:
                    continue  # Will draw these in orange later
                draw_box(img, det_box, (0, 0, 255), thickness=1)

            # Draw under-segmentation cases in ORANGE (thick)
            for det_idx in under_seg_detections:
                gt_indices = under_seg_gt_groups[det_idx]
                label = f"Under-seg: {len(gt_indices)} objects"
                draw_box(img, detected_boxes[det_idx], (0, 165, 255), thickness=4, label=label)

                # Draw the GT boxes it covers in green with thick border
                for gt_idx in gt_indices:
                    draw_box(img, gt_boxes[gt_idx], (0, 255, 0), thickness=3)

            # Draw over-segmentation cases in BLUE (thick)
            for gt_idx in over_seg_gts:
                det_indices = over_seg_det_groups[gt_idx]
                label = f"Over-seg: {len(det_indices)} dets"
                draw_box(img, gt_boxes[gt_idx], (255, 0, 0), thickness=4, label=label)

                # Draw the detections in red with thick border
                for det_idx in det_indices:
                    draw_box(img, detected_boxes[det_idx], (0, 0, 255), thickness=3)

            # Add legend
            legend_y = 30
            cv2.putText(img, "Green: GT boxes", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, "Red: Detections", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(img, "Orange: Under-seg (multi GT -> 1 det)", (10, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(img, "Blue: Over-seg (1 GT -> multi det)", (10, legend_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Determine output subfolder and save images
            # Use basename to avoid subdirectory issues in filename
            base_filename = os.path.basename(img_info['file_name'])

            if len(under_seg_detections) > 0:
                under_seg_images.add(img_info['file_name'])
                under_seg_count += 1
                if under_seg_count <= max_images_per_type:
                    save_dir = os.path.join(output_dir, 'under_segmentation')
                    os.makedirs(save_dir, exist_ok=True)
                    output_path = os.path.join(save_dir, base_filename)
                    success = cv2.imwrite(output_path, img)
                    if not success:
                        print(f"WARNING: Failed to save {output_path}")

            if len(over_seg_gts) > 0:
                over_seg_images.add(img_info['file_name'])
                over_seg_count += 1
                if over_seg_count <= max_images_per_type:
                    save_dir = os.path.join(output_dir, 'over_segmentation')
                    os.makedirs(save_dir, exist_ok=True)
                    output_path = os.path.join(save_dir, base_filename)
                    success = cv2.imwrite(output_path, img)
                    if not success:
                        print(f"WARNING: Failed to save {output_path}")

            # Save combined issues (save all, no limit)
            if len(under_seg_detections) > 0 and len(over_seg_gts) > 0:
                save_dir = os.path.join(output_dir, 'both_issues')
                os.makedirs(save_dir, exist_ok=True)
                output_path = os.path.join(save_dir, base_filename)
                success = cv2.imwrite(output_path, img)
                if not success:
                    print(f"WARNING: Failed to save {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nTotal images with under-segmentation: {len(under_seg_images)}")
    print(f"Total images with over-segmentation: {len(over_seg_images)}")

    # Calculate how many were actually saved
    saved_under = min(under_seg_count, max_images_per_type)
    saved_over = min(over_seg_count, max_images_per_type)

    print(f"\nImages saved:")
    print(f"  - under_segmentation/: {saved_under} of {len(under_seg_images)} (limit: {max_images_per_type})")
    print(f"  - over_segmentation/: {saved_over} of {len(over_seg_images)} (limit: {max_images_per_type})")
    print(f"\nVisualized images saved to: {output_dir}/")
    print(f"\nNote: Use --max-images N to save more images (default is 20)")
    print(f"      Example: --max-images 999999 to save all")
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Multi-Object Detection Issues")
    parser.add_argument('--yolo-model', type=str,
                       default='yolo_cascade_training/yolo_superclass_plankton/weights/best.pt',
                       help='Path to trained YOLO model')
    parser.add_argument('--images', type=str, default='StudyCase',
                       help='Path to images directory')
    parser.add_argument('--annotations', type=str, default='StudyCase/_annotations.coco.json',
                       help='Path to COCO annotations')
    parser.add_argument('--output', type=str, default='overlap_analysis_results/visualized_issues',
                       help='Output directory for visualized images')
    parser.add_argument('--conf', type=float, default=0.15,
                       help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='YOLO NMS IoU threshold')
    parser.add_argument('--match-iou', type=float, default=0.5,
                       help='IoU threshold for matching detection to ground truth')
    parser.add_argument('--coverage', type=float, default=0.7,
                       help='Coverage threshold to consider objects as merged')
    parser.add_argument('--max-images', type=int, default=20,
                       help='Maximum images to save per issue type')

    args = parser.parse_args()

    visualize_overlap_issues(
        yolo_model_path=args.yolo_model,
        images_dir=args.images,
        annotation_path=args.annotations,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        match_iou_threshold=args.match_iou,
        coverage_threshold=args.coverage,
        max_images_per_type=args.max_images
    )
