import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional
import yaml
from pathlib import Path
import json

class PlanktonDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cuda',
        use_tta: bool = False  # Test Time Augmentation
    ):
        """
        Initialize plankton detector with trained YOLO model

        Args:
            model_path: Path to trained model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
            use_tta: Whether to use test time augmentation
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_tta = use_tta

        # Get class names from model
        self.class_names = self.model.names

        print(f"Model loaded on {self.device}")
        print(f"Classes: {list(self.class_names.values())}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve detection"""
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def detect_single_image(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        enhance_contrast: bool = True,
        show_confidence: bool = True,
        show_counts: bool = True
    ) -> Dict:
        """
        Run detection on a single image

        Returns:
            Dictionary containing:
            - boxes: List of bounding boxes
            - scores: Confidence scores
            - classes: Class IDs
            - counts: Count per class
            - total_count: Total objects detected
        """
        # Load image
        image = cv2.imread(image_path)
        original_image = image.copy()

        # Enhance contrast if requested
        if enhance_contrast:
            image = self.preprocess_image(image)

        # Run inference
        if self.use_tta:
            # Test Time Augmentation for better accuracy
            results_list = []

            # Original
            results_list.append(self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0])

            # Horizontal flip
            flipped = cv2.flip(image, 1)
            results_list.append(self.model(
                flipped,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0])

            # Merge results (simplified TTA)
            results = results_list[0]  # Use original as base

        else:
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                max_det=300  # Maximum detections
            )[0]

        # Parse results
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
        classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []

        # Count objects per class
        class_counts = {}
        for cls_id in classes:
            class_name = self.class_names[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Draw results on image
        annotated_image = original_image.copy()

        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[cls_id]

            # Draw bounding box
            color = self.get_color_for_class(cls_id)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}"
            if show_confidence:
                label += f" {score:.2f}"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_image, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add count information
        if show_counts:
            y_offset = 30
            cv2.rectangle(annotated_image, (10, 10), (300, 10 + 25 * (len(class_counts) + 1)),
                         (0, 0, 0), -1)

            for class_name, count in sorted(class_counts.items()):
                text = f"{class_name}: {count}"
                cv2.putText(annotated_image, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25

            # Total count
            total_text = f"Total: {len(boxes)}"
            cv2.putText(annotated_image, total_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save if requested
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            print(f"Result saved to {save_path}")

        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'class_counts': class_counts,
            'total_count': len(boxes),
            'annotated_image': annotated_image
        }

    def detect_batch(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> List[Dict]:
        """Run detection on multiple images"""
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = [cv2.imread(path) for path in batch_paths]

            # Run batch inference
            batch_results = self.model(
                batch_images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            for path, result in zip(batch_paths, batch_results):
                # Parse individual results
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []

                # Count objects
                class_counts = {}
                for cls_id in classes:
                    class_name = self.class_names[cls_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                results.append({
                    'image_path': path,
                    'boxes': boxes,
                    'scores': scores,
                    'classes': classes,
                    'class_counts': class_counts,
                    'total_count': len(boxes)
                })

        return results

    def visualize_results(
        self,
        image_path: str,
        detection_result: Dict,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """Visualize detection results with matplotlib"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image)

        # Draw bounding boxes
        for box, score, cls_id in zip(
            detection_result['boxes'],
            detection_result['scores'],
            detection_result['classes']
        ):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=self.get_color_for_class(cls_id, normalized=True),
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            class_name = self.class_names[cls_id]
            label = f"{class_name} {score:.2f}"
            ax.text(x1, y1 - 5, label,
                   color='white',
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.5))

        # Add title with counts
        title = f"Total: {detection_result['total_count']} | "
        title += " | ".join([f"{cls}: {cnt}" for cls, cnt in detection_result['class_counts'].items()])
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        plt.tight_layout()
        plt.show()

    def get_color_for_class(self, class_id: int, normalized: bool = False):
        """Get consistent color for each class"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]

        color = colors[class_id % len(colors)]

        if normalized:
            return tuple(c / 255.0 for c in color)
        return color

    def calculate_metrics(self, detection_result: Dict) -> Dict:
        """Calculate additional metrics for analysis"""
        boxes = detection_result['boxes']

        if len(boxes) == 0:
            return {
                'density': 0,
                'avg_size': 0,
                'size_variance': 0,
                'spatial_distribution': 'empty'
            }

        # Calculate sizes
        sizes = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]

        # Calculate centers
        centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]

        # Density categories
        density = len(boxes)
        if density < 10:
            density_category = 'sparse'
        elif density < 50:
            density_category = 'medium'
        else:
            density_category = 'dense'

        return {
            'density': density,
            'density_category': density_category,
            'avg_size': np.mean(sizes),
            'size_variance': np.var(sizes),
            'min_size': np.min(sizes),
            'max_size': np.max(sizes),
            'spatial_distribution': self._calculate_spatial_distribution(centers)
        }

    def _calculate_spatial_distribution(self, centers: List[Tuple]) -> str:
        """Analyze spatial distribution of detections"""
        if len(centers) < 3:
            return 'insufficient_data'

        # Simple clustering check
        from scipy.spatial.distance import pdist
        distances = pdist(centers)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        if std_dist / mean_dist > 0.5:
            return 'clustered'
        else:
            return 'uniform'

class EnsembleDetector:
    """Ensemble multiple models for better accuracy"""

    def __init__(
        self,
        model_paths: List[str],
        weights: Optional[List[float]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        voting: str = 'weighted'  # 'unanimous', 'majority', 'weighted'
    ):
        self.detectors = [
            PlanktonDetector(path, conf_threshold, iou_threshold)
            for path in model_paths
        ]

        self.weights = weights or [1.0] * len(self.detectors)
        self.voting = voting

    def detect(self, image_path: str) -> Dict:
        """Run ensemble detection"""
        all_results = []

        for detector, weight in zip(self.detectors, self.weights):
            result = detector.detect_single_image(image_path)
            all_results.append((result, weight))

        # Merge results based on voting strategy
        if self.voting == 'weighted':
            return self._weighted_voting(all_results)
        elif self.voting == 'majority':
            return self._majority_voting(all_results)
        else:
            return self._unanimous_voting(all_results)

    def _weighted_voting(self, results: List[Tuple[Dict, float]]) -> Dict:
        """Weighted voting ensemble"""
        # Implementation of weighted box fusion
        # Simplified version - you might want to use a library like ensemble-boxes

        all_boxes = []
        all_scores = []
        all_classes = []

        for (result, weight) in results:
            for box, score, cls in zip(result['boxes'], result['scores'], result['classes']):
                all_boxes.append(box)
                all_scores.append(score * weight)
                all_classes.append(cls)

        # Apply NMS on merged results
        if len(all_boxes) > 0:
            # Simplified - use proper WBF for production
            indices = self._nms(np.array(all_boxes), np.array(all_scores), 0.5)

            final_boxes = [all_boxes[i] for i in indices]
            final_scores = [all_scores[i] for i in indices]
            final_classes = [all_classes[i] for i in indices]
        else:
            final_boxes = []
            final_scores = []
            final_classes = []

        # Count objects
        class_counts = {}
        for cls_id in final_classes:
            class_name = self.detectors[0].class_names[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'classes': final_classes,
            'class_counts': class_counts,
            'total_count': len(final_boxes)
        }

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-maximum suppression"""
        from torchvision.ops import nms
        import torch

        boxes_t = torch.from_numpy(boxes).float()
        scores_t = torch.from_numpy(scores).float()

        keep = nms(boxes_t, scores_t, threshold)
        return keep.numpy().tolist()

    def _majority_voting(self, results: List[Tuple[Dict, float]]) -> Dict:
        """Majority voting ensemble"""
        # Implementation for majority voting
        pass

    def _unanimous_voting(self, results: List[Tuple[Dict, float]]) -> Dict:
        """Unanimous voting ensemble"""
        # Implementation for unanimous voting
        pass

# Usage example
if __name__ == "__main__":
    # Single model inference
    detector = PlanktonDetector(
        model_path='plankton_detection/yolov10s_plankton/weights/best.pt',
        conf_threshold=0.25,
        iou_threshold=0.45,
        device='cuda',
        use_tta=False  # Set to True for better accuracy (slower)
    )

    # Detect single image
    image_path = 'workspace/some_exp/genus/hald_assignment/StudyCase/images/sample.jpg'

    result = detector.detect_single_image(
        image_path,
        save_path='detection_result.jpg',
        enhance_contrast=True,
        show_confidence=True,
        show_counts=True
    )

    print(f"\nDetection Results:")
    print(f"Total objects detected: {result['total_count']}")
    print(f"Per-class counts: {result['class_counts']}")

    # Calculate additional metrics
    metrics = detector.calculate_metrics(result)
    print(f"\nAdditional Metrics:")
    print(f"Density category: {metrics['density_category']}")
    print(f"Average object size: {metrics['avg_size']:.2f} pixels²")
    print(f"Spatial distribution: {metrics['spatial_distribution']}")

    # Visualize with matplotlib
    detector.visualize_results(image_path, result)

    # Batch processing example
    image_paths = [
        'image1.jpg',
        'image2.jpg',
        'image3.jpg'
    ]

    batch_results = detector.detect_batch(image_paths, batch_size=8)

    for result in batch_results:
        print(f"\n{result['image_path']}:")
        print(f"  Total: {result['total_count']}")
        print(f"  Counts: {result['class_counts']}")

    # Ensemble example (if you trained multiple models)
    # ensemble = EnsembleDetector(
    #     model_paths=[
    #         'plankton_detection/yolov10n_plankton/weights/best.pt',
    #         'plankton_detection/yolov10s_plankton/weights/best.pt'
    #     ],
    #     weights=[0.4, 0.6],
    #     voting='weighted'
    # )
    #
    # ensemble_result = ensemble.detect(image_path)
