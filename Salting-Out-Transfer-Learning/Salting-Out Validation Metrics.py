import os
import numpy as np
import json
import random
import sys
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import zeros, asarray
import pycocotools.mask as mask_util

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mrcnn
import mrcnn.utils
import mrcnn.config
import mrcnn.model
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
from mrcnn.utils import compute_ap
from typing import List, Any

# Define for pylance
AP: float
image: np.ndarray
image_meta: Any
precisions_at_t: List[float]
recalls_at_t: List[float]
gt_class_id: np.ndarray
gt_bbox: np.ndarray
gt_mask: np.ndarray
overlaps: np.ndarray

# Define the class names (Background + Reacting + Reaction Complete)
CLASS_NAMES = ['BG', 'Reacting', 'Reaction Complete']

class SaltingOutDataset(mrcnn.utils.Dataset):
    def load_dataset(self, annotation_file, split_ratio=0.8, is_train=True):
        """Load dataset from COCO 1.0 annotations."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Add classes from COCO categories
        for category in data['categories']:
            self.add_class("dataset", category['id'], category['name'])

        # Split data into training and validation
        image_ids = list(data['images'])
        random.shuffle(image_ids)
        split_index = int(len(image_ids) * split_ratio)

        if is_train:
            selected_images = image_ids[:split_index]
        else:
            selected_images = image_ids[split_index:]

        print(f"Annotation file: {annotation_file}")
        print(f"Total images: {len(image_ids)}, Train: {len(selected_images)}" if is_train else f"Val: {len(selected_images)}")

        for image_info in selected_images:
            image_id = image_info['id']
            annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

            if len(annotations) == 0:
                print(f"Skipping {image_info['file_name']} - No annotations found.")
                continue

            image_path = os.path.join(os.path.dirname(annotation_file), image_info['file_name'])
            if not os.path.exists(image_path):
                print(f"Skipping {image_info['file_name']} - File not found.")
                continue

            self.add_image(
                source="dataset",
                image_id=image_id,
                path=image_path,
                width=image_info['width'],
                height=image_info['height'],
                annotations=annotations
            )

        print(f"Images loaded for {'train' if is_train else 'validation'}: {len(self.image_info)}")

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        masks = []
        class_ids = []

        for annotation in annotations:
            # Get segmentation and category_id
            segmentation = annotation.get('segmentation', None)
            category_id = annotation.get('category_id', None)

            if segmentation:
                if isinstance(segmentation, dict):  # RLE (Run-Length Encoding)
                    rle = mask_util.frPyObjects(segmentation, info['height'], info['width'])
                    mask = mask_util.decode(rle)
                elif isinstance(segmentation, list):  # Polygon
                    mask = np.zeros((info['height'], info['width']), dtype=np.uint8)
                    for poly in segmentation:
                        poly = np.array(poly).reshape((-1, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                else:
                    print(f"Unknown segmentation format for annotation ID {annotation['id']}")
                    continue
            else:
                # Skip annotations with no segmentation
                continue

            # Add the mask and class ID
            masks.append(mask)
            class_ids.append(self.map_source_class_id("dataset.{}".format(category_id)))

        if masks:
            masks = np.stack(masks, axis=-1)
        else:
            # Return an empty mask and class_ids if no valid segmentations are found
            masks = np.empty((info['height'], info['width'], 0), dtype=np.uint8)

        return masks, np.array(class_ids, dtype=np.int32)

    def ann_to_mask(self, polygons, height, width):
        """Convert polygons to a binary mask."""
        mask = zeros((height, width), dtype='uint8')
        for polygon in polygons:
            coords = asarray(polygon).reshape((-1, 2))
            rr, cc = mrcnn.utils.polygon(coords[:, 1], coords[:, 0], (height, width))
            mask[rr, cc] = 1
        return mask

class SaltingOutInferenceConfig(Config):
    NAME = "salting_out_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)

inference_model = MaskRCNN(
    mode="inference",
    config=SaltingOutInferenceConfig(),
    model_dir=os.getcwd()
)

inference_model.load_weights(
    filepath='salting_out_heads_trained_3.h5', 
    by_name=True
)

def calculate_precision_recall_f1(tp, fp, fn):
    """Calculate precision, recall, and F1 score."""
    # Convert to floats to avoid integer division
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)

    # Compute precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Compute recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Compute F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Unpack coordinates
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    # Calculate intersection
    inter_y_min = max(y1_min, y2_min)
    inter_x_min = max(x1_min, x2_min)
    inter_y_max = min(y1_max, y2_max)
    inter_x_max = min(x1_max, x2_max)

    inter_area = max(0, inter_y_max - inter_y_min) * max(0, inter_x_max - inter_x_min)

    # Calculate union
    box1_area = (y1_max - y1_min) * (x1_max - x1_min)
    box2_area = (y2_max - y2_min) * (x2_max - x2_min)
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0

    # IoU calculation
    return inter_area / union_area

def calculate_mean_iou(gt_boxes, pred_boxes):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0

    ious = []
    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in pred_boxes:
            iou = calculate_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
        ious.append(best_iou)
    return np.mean(ious) if ious else 0

def global_coco_evaluation_with_visualization(datasets, model, config):
    global_tp, global_fp, global_fn = 0, 0, 0
    global_ious = []
    thresholds = np.round(np.arange(0.50, 1.00, 0.05), 2)  # Round to 2 decimal places
    global_aps = {t: [] for t in thresholds}
    iou_per_image = []

    # For thresholds >0.50 and >0.75
    tp_gt_50, fp_gt_50, fn_gt_50 = 0, 0, 0
    tp_gt_75, fp_gt_75, fn_gt_75 = 0, 0, 0
    ious_gt_50 = []
    ious_gt_75 = []

    # Inside global_coco_evaluation_with_visualization
    for dataset in datasets:
        for image_id in tqdm(dataset.image_ids, desc="Validating", unit="image"):
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
                dataset, config, image_id, use_mini_mask=False
            )
            molded_image = mold_image(image, config)
            molded_images = np.expand_dims(molded_image, axis=0)

            results = model.detect(molded_images, verbose=0)[0]

            rois = results['rois']
            class_ids = results['class_ids']
            scores = results['scores']
            masks = results['masks']

            if len(rois) == 0 or len(gt_bbox) == 0:
                continue

            # Track global TP, FP, and FN
            for t in thresholds:
                rounded_t = round(t, 2)
                AP, precisions_at_t, recalls_at_t, overlaps = compute_ap(
                    gt_bbox, gt_class_id, gt_mask, rois, class_ids, scores, masks, iou_threshold=rounded_t
                )
                global_aps[rounded_t].append(AP)

                # Initialize matched indices
                matched_gt = set()
                matched_pred = set()

                # Match predictions to ground truth
                for i, gt_box in enumerate(gt_bbox):
                    for j, pred_box in enumerate(rois):
                        if j in matched_pred:
                            continue  # Skip already matched predictions
                        iou = calculate_iou(gt_box, pred_box)
                        if iou >= rounded_t:
                            matched_gt.add(i)
                            matched_pred.add(j)

                # Count true positives, false negatives, and false positives
                tp = len(matched_gt)
                fn = len(gt_bbox) - tp
                fp = len(rois) - len(matched_pred)

                global_tp += tp
                global_fp += fp
                global_fn += fn

            # Update IoU metrics
            mean_iou = calculate_mean_iou(gt_bbox, rois)
            global_ious.append(mean_iou)
            iou_per_image.append(mean_iou)

            # For IoU >0.50 and >0.75 calculations
            matches_gt_50 = set()
            matches_gt_75 = set()

            for i, gt_box in enumerate(gt_bbox):
                best_iou = 0
                best_idx = -1
                for j, pred_box in enumerate(rois):
                    iou = calculate_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

                if best_idx != -1:
                    # IoU > 0.50
                    if best_iou >= 0.50:
                        tp_gt_50 += 1
                        matches_gt_50.add(best_idx)
                        ious_gt_50.append(best_iou)

                    # IoU > 0.75
                    if best_iou >= 0.75:
                        tp_gt_75 += 1
                        matches_gt_75.add(best_idx)
                        ious_gt_75.append(best_iou)

            fp_gt_50 += len(rois) - len(matches_gt_50)
            fn_gt_50 += len(gt_bbox) - len(matches_gt_50)

            fp_gt_75 += len(rois) - len(matches_gt_75)
            fn_gt_75 += len(gt_bbox) - len(matches_gt_75)

    # Global metrics aggregation after the loop
    precision, recall, f1_score = calculate_precision_recall_f1(global_tp, global_fp, global_fn)

    # Fixing IoU-specific metrics aggregation
    precision_gt_50, recall_gt_50, f1_gt_50 = calculate_precision_recall_f1(tp_gt_50, fp_gt_50, fn_gt_50)
    precision_gt_75, recall_gt_75, f1_gt_75 = calculate_precision_recall_f1(tp_gt_75, fp_gt_75, fn_gt_75)

    mean_iou = np.mean(global_ious) if global_ious else 0
    mean_iou_gt_50 = np.mean(ious_gt_50) if ious_gt_50 else 0
    mean_iou_gt_75 = np.mean(ious_gt_75) if ious_gt_75 else 0

    overall_map = np.mean([np.mean(global_aps[t]) for t in thresholds if global_aps[t]]) if any(global_aps[t] for t in thresholds) else 0  # Ensure non-empty
    map_gt_50 = np.mean(global_aps[0.50]) if global_aps[0.50] else 0
    map_gt_75 = np.mean(global_aps[0.75]) if global_aps[0.75] else 0

    # Compute and display per-threshold mAP
    map_per_iou = {t: np.mean(global_aps[t]) for t in thresholds}
    print("\nDetailed mAP per IoU threshold:")
    for t, mean_ap_t in map_per_iou.items():
        print(f"IoU Threshold {t:.2f}: Mean AP = {mean_ap_t:.4f}")

    # Display aggregated metrics
    print(f"\nOverall mAP@[IoU=0.50:0.95]: {overall_map:.4f}")
    print(f"Mean Precision: {precision:.4f}")
    print(f"Mean Recall: {recall:.4f}")
    print(f"Mean F1-Score: {f1_score:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")

    # IoU > 0.50
    print("\nFor IoU > 0.50:")
    print(f"Precision: {precision_gt_50:.4f}")
    print(f"Recall: {recall_gt_50:.4f}")
    print(f"F1-Score: {f1_gt_50:.4f}")
    print(f"Mean IoU: {mean_iou_gt_50:.4f}")
    print(f"mAP: {map_gt_50:.4f}")

    # IoU > 0.75
    print("\nFor IoU > 0.75:")
    print(f"Precision: {precision_gt_75:.4f}")
    print(f"Recall: {recall_gt_75:.4f}")
    print(f"F1-Score: {f1_gt_75:.4f}")
    print(f"Mean IoU: {mean_iou_gt_75:.4f}")
    print(f"mAP: {map_gt_75:.4f}")

    # Visualization: IoU vs Images
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(iou_per_image) + 1), iou_per_image, marker="o", label="IoU per Image")
    plt.title("IoU vs Image Index")
    plt.xlabel("Image Index")
    plt.ylabel("Mean IoU")
    plt.axhline(y=mean_iou, color="r", linestyle="--", label=f"Mean IoU = {mean_iou:.4f}")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Visualization: IoU vs mAP
    plt.figure(figsize=(10, 6))
    plt.plot(list(map_per_iou.keys()), list(map_per_iou.values()), marker="o", label="IoU vs mAP")
    plt.title("IoU vs mAP")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Mean Average Precision (mAP)")
    plt.xticks(thresholds)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Return detailed metrics
    return {
        "overall_map": overall_map,
        "map_per_iou": map_per_iou,
        "mean_precision": precision,
        "mean_recall": recall,
        "mean_f1_score": f1_score,
        "mean_iou": mean_iou,
        "precision_gt_50": precision_gt_50,
        "recall_gt_50": recall_gt_50,
        "f1_gt_50": f1_gt_50,
        "mean_iou_gt_50": mean_iou_gt_50,
        "map_gt_50": map_gt_50,
        "precision_gt_75": precision_gt_75,
        "recall_gt_75": recall_gt_75,
        "f1_gt_75": f1_gt_75,
        "mean_iou_gt_75": mean_iou_gt_75,
        "map_gt_75": map_gt_75,
    }

if __name__ == "__main__":
    validation_dataset = SaltingOutDataset()

    annotation_files = [
        "dataset.json",
    ]
    datasets = []
    for annotation_file in annotation_files:
        dataset = SaltingOutDataset()
        dataset.load_dataset(annotation_file=annotation_file, split_ratio=0.8, is_train=False)
        dataset.prepare()
        datasets.append(dataset)
metrics = global_coco_evaluation_with_visualization(datasets, inference_model, SaltingOutInferenceConfig())
