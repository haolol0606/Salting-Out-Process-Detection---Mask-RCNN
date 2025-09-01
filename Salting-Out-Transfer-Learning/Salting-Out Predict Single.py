import sys
sys.path.append('C:/Users/user/Documents/Degree Note File (XMUM)/Year 4 Sem 1/Thesis 2/Code/Mask-RCNN-TF2-Python3.7.3')

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

# Define the class names (Background + Reacting + Reaction Complete)
CLASS_NAMES = ['BG', 'Reacting', 'Reaction Complete']

class SaltingOutConfig(mrcnn.config.Config):
    NAME = "salting_out_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3  # Background + Reacting + Reaction Complete
    STEPS_PER_EPOCH = 131
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
inference_model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SaltingOutConfig(),
                             model_dir=os.getcwd())

# Load the trained weights into the inference model
inference_model.load_weights(
    filepath=r'C:\Users\user\Documents\Degree Note File (XMUM)\Year 4 Sem 1\Thesis 2\Code\Mask-RCNN-TF2-Python3.7.3\Salting-Out-Transfer-Learning\log\salting_out_heads_trained_3.h5', 
    by_name=True
)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread(r"C:\Users\user\Documents\Degree Note File (XMUM)\Year 4 Sem 1\Thesis 2\Testing 13 Jan\DATA431.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = inference_model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
