import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from prostate import prostate_train


# Directory to save logs and trained model
MODEL_DIR = '../logs/prostate20190201T1317/'

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
Prostate_WEIGHTS_PATH = "mask_rcnn_prostate_0066.h5"  # TODO: update this path

config = prostate_train.ProstateConfig()
Prostate_DIR = '../mask_rcnn_3T_zone'

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

dataset = prostate_train.ProstateDataset()
dataset.load_prostate(Prostate_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

weights_path = MODEL_DIR+Prostate_WEIGHTS_PATH

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


image_ids_list=dataset.image_ids.tolist()

for image_id in image_ids_list:
    # image_id = random.choice(dataset.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))

    # Run object detection
    results = model.detect([image], verbose=1)
    ax = get_ax(1)
    r = results[0]
   # print(len(results[0]['class_ids']))
    index_pz=None
    index_cg=None
    if len(results[0]['rois']) > 1:
        for i in range(len(results[0]['class_ids'])):
            if r['class_ids'][i]==1:
                if index_pz==None:
                    index_pz=i
                else:
                    continue

            if r['class_ids'][i]==2:
                if index_cg==None:
                    index_cg=i
                else:
                    continue
        if (index_pz!=None) and (index_cg!=None):
            print('status:',1)
            r['rois']=np.stack((results[0]['rois'][index_pz],results[0]['rois'][index_cg]))
            r['masks']=np.stack((results[0]['masks'][:,:,index_pz],results[0]['masks'][:,:,index_cg]),axis=2)
            r['class_ids']=np.stack((results[0]['class_ids'][index_pz],results[0]['class_ids'][index_cg]))
            r['scores']=np.stack((results[0]['scores'][index_pz],results[0]['scores'][index_cg]))

        else:
            print('status:',2)
            r['rois']=results[0]['rois'][0].reshape(1,4)
            r['masks']=results[0]['masks'][:,:,0].reshape(512,512,1)
            r['class_ids']=np.array([results[0]['class_ids'][0]])
            r['scores']=np.array([results[0]['scores'][0]])
            print(r['rois'].shape,r['masks'].shape,r['class_ids'].shape,r['scores'].shape)
        
        ax = get_ax(1)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        dataset.class_names, r['scores'], ax=ax,
                                        title=info["id"])
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
    else:
        print('status:',3)
       # ax = get_ax(1)
        r=results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax,
                                title=info["id"])
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)

'''
    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax,
                                title=info["id"])
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
'''
