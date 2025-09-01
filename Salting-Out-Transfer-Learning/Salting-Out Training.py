import os
import json
import random
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping
import pycocotools.mask as mask_util
import numpy as np
import cv2
from numpy import zeros, asarray

#import mrcnn
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mrcnn.utils
import mrcnn.config
import mrcnn.model
import imgaug  # For data augmentation

# Check GPU Availability
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", [device.name for device in device_lib.list_local_devices() if device.device_type == 'GPU'])
gpus = [device for device in device_lib.list_local_devices() if device.device_type == 'GPU']
if gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace "0" with the GPU ID you want to use
    print(f"Using GPU: {gpus[0].name}")
else:
    print("No GPU found. Running on CPU.")

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

        # Add images and annotations
        for image_info in selected_images:
            image_id = image_info['id']
            annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

            # Filter out images without annotations
            if len(annotations) == 0:
                continue

            image_path = os.path.join(os.path.dirname(annotation_file), image_info['file_name'])

            if not os.path.exists(image_path):
                print(f"Warning: File not found {image_path}. Skipping.")
                continue

            self.add_image(
                source="dataset",
                image_id=image_id,
                path=image_path,
                width=image_info['width'],
                height=image_info['height'],
                annotations=annotations
            )

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

class SaltingOutConfig(mrcnn.config.Config):
    NAME = "salting_out_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + Reacting + Reaction Complete
    STEPS_PER_EPOCH = None
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    def update_steps_per_epoch(self, train_dataset_size):
        """Update STEPS_PER_EPOCH based on the training dataset size."""
        self.STEPS_PER_EPOCH = int(train_dataset_size / (self.IMAGES_PER_GPU * self.GPU_COUNT))

# Define the file paths for datasets
annotation_files = [
    "datasets.json"
]

# Initialize datasets
train_dataset = SaltingOutDataset()
validation_dataset = SaltingOutDataset()

# Load datasets
for annotation_file in annotation_files:
    train_dataset.load_dataset(annotation_file=annotation_file, split_ratio=0.8, is_train=True)
    validation_dataset.load_dataset(annotation_file=annotation_file, split_ratio=0.8, is_train=False)

train_dataset.prepare()
validation_dataset.prepare()

train_image_count = len(train_dataset.image_ids)
validation_image_count = len(validation_dataset.image_ids)

# Print dataset statistics
print(f"Number of training images: {train_image_count}")
print(f"Number of validation images: {validation_image_count}")

# Suppress unnecessary TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Define the log directory path consistently
log_train_dir = "\log"
os.makedirs(log_train_dir, exist_ok=True)

# Model Configuration
saltingout_config = SaltingOutConfig()
saltingout_config.update_steps_per_epoch(train_image_count)

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir=log_train_dir, 
                             config=saltingout_config)

# Load weights with the correct path formatting
model.load_weights(filepath="salting_out_heads_trained_3.h5",
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Data Augmentation: Define augmentation strategy
augmentation = imgaug.augmenters.Sequential([
    imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Fliplr(1.0)),  # 50% chance of horizontal flip
    imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Affine(scale=(0.8, 1.2))),  # 50% chance of scaling
    imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Rotate((-15, 15))),  # 50% chance of rotation
    imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Multiply((0.8, 1.2)))  # 50% chance of brightness change
])

# Define Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Stop after 5 epochs of no improvement
    restore_best_weights=True  # Restore best weights at the end
)

# Train the Mask R-CNN model
model.train(
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
    learning_rate=saltingout_config.LEARNING_RATE/10,
    epochs=50,
    layers='all',
    augmentation=augmentation,
    custom_callbacks=[early_stopping],
)

# Save the final trained weights
model_path = os.path.join(log_train_dir, 'salting_out_heads_trained_3.h5')
model.keras_model.save_weights(model_path)
print(f"Final weights saved to {model_path}")
