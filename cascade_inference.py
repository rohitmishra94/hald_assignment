import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import ArcFace model (assuming it's in evaluation/)
import sys
sys.path.append('evaluation')
from model_arc import EmbeddingModel

class CascadePlanktonIdentifier:
    def __init__(
        self,
        yolo_model_path: str,
        arcface_model_path: str,
        prototypes_path: str,
        class_mapping_path: str,
        device: str = 'cuda',
        yolo_conf_threshold: float = 0.15,  # Lower for high recall
        yolo_iou_threshold: float = 0.5,
        arcface_threshold: float = 0.6,  # Cosine similarity threshold
        target_size: Tuple[int, int] = (128, 128)
    ):
        """
        2-Stage Cascade System: YOLO Detection + ArcFace Identification

        Stage 1 (YOLO): Detect all plankton objects (high recall)
        Stage 2 (ArcFace): Identify species using metric learning

        Args:
            yolo_model_path: Path to trained YOLO model (super-class detector)
            arcface_model_path: Path to trained ArcFace model
            prototypes_path: Path to class prototypes (saved embeddings)
            class_mapping_path: Path to class to index mapping
            device: Device for inference
            yolo_conf_threshold: YOLO confidence (low for high recall)
            yolo_iou_threshold: YOLO IoU threshold for NMS
            arcface_threshold: ArcFace cosine similarity threshold
            target_size: Resize size for ArcFace (128x128)
        """

        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Initializing Cascade System on {self.device}...")

        # Stage 1: Load YOLO detector
        print("Loading YOLO detector...")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_conf = yolo_conf_threshold
        self.yolo_iou = yolo_iou_threshold

        # Stage 2: Load ArcFace identifier
        print("Loading ArcFace identifier...")
        self.arcface_model = EmbeddingModel(embedding_size=512, pretrained=False)
        self.arcface_model.load_state_dict(torch.load(arcface_model_path, map_location=self.device))
        self.arcface_model.to(self.device)
        self.arcface_model.eval()

        # Load class prototypes
        print("Loading class prototypes...")
        self.prototypes = torch.load(prototypes_path, map_location=self.device)
        self.prototype_embeddings = self.prototypes['embeddings']  # [num_classes, 512]
        self.prototype_labels = self.prototypes['labels']  # List of class names

        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        self.idx_to_class = {int(k): v for k, v in self.class_mapping['idx_to_class'].items()}

        self.arcface_threshold = arcface_threshold
        self.target_size = target_size

        print(f"System initialized!")
        print(f"  YOLO confidence threshold: {self.yolo_conf}")
        print(f"  ArcFace similarity threshold: {self.arcface_threshold}")
        print(f"  Number of classes: {len(self.prototype_labels)}")

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Stage 1: Detect plankton objects using YOLO

        Returns list of detected objects with bounding boxes
        """
        results = self.yolo_model(
            image,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            verbose=False
        )[0]

        detections = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)

                # Crop detected region
                cropped = image[y1:y2, x1:x2]

                if cropped.size > 0:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'yolo_confidence': float(score),
                        'cropped_image': cropped
                    })

        return detections

    def identify_object(self, cropped_image: np.ndarray) -> Dict:
        """
        Stage 2: Identify species using ArcFace

        Returns predicted class and confidence
        """
        # Preprocess image
        resized = cv2.resize(cropped_image, self.target_size)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        image_tensor = torch.from_numpy(resized_rgb).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]

        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Extract embedding
        with torch.no_grad():
            embedding = self.arcface_model(image_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize

        # Calculate cosine similarity with all prototypes
        similarities = F.cosine_similarity(
            embedding,
            self.prototype_embeddings,
            dim=1
        )

        # Get top prediction
        max_similarity, predicted_idx = torch.max(similarities, dim=0)

        predicted_class = self.prototype_labels[predicted_idx.item()]
        confidence = max_similarity.item()

        # Get top-5 predictions
        top5_similarities, top5_indices = torch.topk(similarities, k=min(5, len(similarities)))
        top5_predictions = [
            {
                'class': self.prototype_labels[idx.item()],
                'confidence': sim.item()
            }
            for idx, sim in zip(top5_indices, top5_similarities)
        ]

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top5_predictions': top5_predictions,
            'is_confident': confidence >= self.arcface_threshold
        }

    def process_image(self, image_path: str) -> Dict:
        """
        Full cascade pipeline: Detect + Identify

        Returns all detections with their identifications
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Stage 1: Detect objects
        detections = self.detect_objects(image)

        # Stage 2: Identify each detected object
        results = []

        for det in detections:
            identification = self.identify_object(det['cropped_image'])

            results.append({
                'bbox': det['bbox'],
                'yolo_confidence': det['yolo_confidence'],
                'predicted_class': identification['predicted_class'],
                'arcface_confidence': identification['confidence'],
                'is_confident': identification['is_confident'],
                'top5_predictions': identification['top5_predictions']
            })

        # Count objects per class
        class_counts = {}
        for res in results:
            if res['is_confident']:  # Only count confident predictions
                cls = res['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1

        return {
            'image_path': image_path,
            'total_detections': len(results),
            'confident_detections': sum(1 for r in results if r['is_confident']),
            'results': results,
            'class_counts': class_counts,
            'image': image
        }

    def visualize_results(
        self,
        result: Dict,
        save_path: Optional[str] = None,
        show_top5: bool = False
    ):
        """
        Visualize detection and identification results
        """
        image = result['image'].copy()

        # Draw bounding boxes and labels
        for res in result['results']:
            x1, y1, x2, y2 = res['bbox']
            predicted_class = res['predicted_class']
            arcface_conf = res['arcface_confidence']
            yolo_conf = res['yolo_confidence']

            # Color based on confidence
            if res['is_confident']:
                color = (0, 255, 0)  # Green for confident
            else:
                color = (0, 165, 255)  # Orange for uncertain

            # Draw bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{predicted_class} ({arcface_conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            cv2.rectangle(image, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add summary
        summary_y = 30
        cv2.rectangle(image, (10, 10),
                     (400, 10 + 25 * (len(result['class_counts']) + 2)),
                     (0, 0, 0), -1)

        cv2.putText(image, f"Total Detected: {result['total_detections']}",
                   (20, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        summary_y += 25

        cv2.putText(image, f"Confident: {result['confident_detections']}",
                   (20, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        summary_y += 25

        for class_name, count in sorted(result['class_counts'].items()):
            cv2.putText(image, f"{class_name}: {count}",
                       (20, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            summary_y += 25

        if save_path:
            cv2.imwrite(save_path, image)
            print(f"Result saved to {save_path}")

        # Convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"Cascade Detection Results - {result['confident_detections']}/{result['total_detections']} confident")
        plt.tight_layout()
        plt.show()

    def batch_process(
        self,
        image_paths: List[str],
        save_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Process multiple images
        """
        from tqdm import tqdm

        all_results = []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.process_image(img_path)
                all_results.append(result)

                if save_dir:
                    img_name = Path(img_path).stem
                    save_path = os.path.join(save_dir, f"{img_name}_result.jpg")
                    self.visualize_results(result, save_path=save_path)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        return all_results

    def generate_report(self, results: List[Dict], output_path: str = 'cascade_report.txt'):
        """
        Generate summary report for batch processing
        """
        total_images = len(results)
        total_detections = sum(r['total_detections'] for r in results)
        total_confident = sum(r['confident_detections'] for r in results)

        # Aggregate class counts
        overall_counts = {}
        for r in results:
            for cls, count in r['class_counts'].items():
                overall_counts[cls] = overall_counts.get(cls, 0) + count

        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CASCADE PIPELINE REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total images processed: {total_images}\n")
            f.write(f"Total objects detected: {total_detections}\n")
            f.write(f"Confident identifications: {total_confident}\n")
            f.write(f"Confidence rate: {total_confident/total_detections*100:.1f}%\n\n")
            f.write("="*60 + "\n")
            f.write("CLASS COUNTS\n")
            f.write("="*60 + "\n")

            for cls, count in sorted(overall_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{cls:<30} {count:>5}\n")

            f.write("="*60 + "\n")

        print(f"Report saved to {output_path}")

# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cascade Plankton Identification")
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--batch', type=str, help='Directory with images for batch processing')
    parser.add_argument('--yolo', type=str, default='yolo_superclass_dataset/runs/detect/train/weights/best.pt')
    parser.add_argument('--arcface', type=str, default='arcface_models/best_model.pth')
    parser.add_argument('--prototypes', type=str, default='arcface_models/class_prototypes.pth')
    parser.add_argument('--mapping', type=str, default='arcface_dataset/class_mapping.json')
    parser.add_argument('--output', type=str, default='cascade_results')
    parser.add_argument('--yolo-conf', type=float, default=0.15)
    parser.add_argument('--arcface-conf', type=float, default=0.6)

    args = parser.parse_args()

    # Initialize cascade system
    cascade = CascadePlanktonIdentifier(
        yolo_model_path=args.yolo,
        arcface_model_path=args.arcface,
        prototypes_path=args.prototypes,
        class_mapping_path=args.mapping,
        yolo_conf_threshold=args.yolo_conf,
        arcface_threshold=args.arcface_conf
    )

    # Process single image
    if args.image:
        result = cascade.process_image(args.image)
        print(f"\nResults:")
        print(f"  Total detections: {result['total_detections']}")
        print(f"  Confident identifications: {result['confident_detections']}")
        print(f"  Class counts: {result['class_counts']}")

        cascade.visualize_results(result, save_path=f"{args.output}/result.jpg")

    # Batch processing
    elif args.batch:
        import glob
        image_paths = glob.glob(os.path.join(args.batch, '*.jpg'))
        image_paths += glob.glob(os.path.join(args.batch, '*.png'))

        print(f"Found {len(image_paths)} images")

        results = cascade.batch_process(image_paths, save_dir=args.output)
        cascade.generate_report(results, f"{args.output}/report.txt")

    else:
        print("Please provide --image or --batch argument")
