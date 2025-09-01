import sys
import cv2
import os
import numpy as np
# Append the required path for Mask R-CNN
sys.path.append('C:/Users/user/Documents/Degree Note File (XMUM)/Year 4 Sem 1/Thesis 2/Code/Mask-RCNN-TF2-Python3.7.3')
from mrcnn import config, model

# Define the class names (Background + Reacting + Reaction Complete)
CLASS_NAMES = ['BG', 'Reacting', 'Reaction Complete']

# Configuration for Salting Out
class SaltingOutConfig(config.Config):
    NAME = "salting_out_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3  # Background + Reacting + Reaction Complete
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

# Initialize the Mask R-CNN model for inference
inference_model = model.MaskRCNN(mode="inference", 
                                 config=SaltingOutConfig(),
                                 model_dir=os.getcwd())

# Load the trained weights
inference_model.load_weights(
    filepath=r'C:\Users\user\Documents\Degree Note File (XMUM)\Year 4 Sem 1\Thesis 2\Code\Mask-RCNN-TF2-Python3.7.3\Salting-Out-Transfer-Learning\log\salting_out_heads_trained2.h5', 
    by_name=True
)

# Open video capture (0 for webcam or specify a video file path)
cap = cv2.VideoCapture(r"C:\Users\user\Documents\Degree Note File (XMUM)\Year 4 Sem 1\Thesis 2\Thesis Dataset 5\example.mp4")  # Use "0" for webcam or replace with video file path

# Check if video capture is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream or file.")
    sys.exit()

# Process each frame in real-time
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error.")
        break

    # Convert frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = inference_model.detect([rgb_frame], verbose=0)
    r = results[0]

    # Draw the annotations on the frame
    for i in range(len(r['rois'])):
        # Extract details for each detected object
        y1, x1, y2, x2 = r['rois'][i]
        class_id = r['class_ids'][i]
        score = r['scores'][i]
        label = CLASS_NAMES[class_id]

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add the label and score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the mask
        mask = r['masks'][:, :, i]
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[mask] = [0, 255, 0]  # Green color for the mask
        frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    # Display the annotated frame
    cv2.imshow('Real-Time Mask R-CNN', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
