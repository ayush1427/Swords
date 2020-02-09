"""
CNN
Train on the original and copied dataset.
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 guns_and_swords.py train --dataset=/path/to/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 guns_and_swords.py train --dataset=/path/to/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 guns_and_swords.py train --dataset=/path/to/dataset --weights=imagenet
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw


'''from keras import backend as K
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #allows dynamic growth
config.gpu_options.visible_device_list = "2" #set GPU number
set_session(tf.Session(config=config))'''


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import CNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class orgforgConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "orgforg"
# We use a GPU with 16GB memory, which can fit three image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 3
# Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + original + copied
# Number of training steps per epoch
    STEPS_PER_EPOCH = 100
# Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class orgforgDataset(utils.Dataset): 

    def load_orgforg(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("orgforg", 1, "`")
        self.add_class("orgforg", 2, "f") 
        
        #self.add_class("orgforg", 1, "forged")
        #self.add_class("orgforg", 2, "original") 
        
        self.class_name_to_ids = {'`':1,'f':2}
       
        #self.class_name_to_ids = {'forged':1,'original':2} 

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            #polygons = [r['shape_attributes'] for r in a['regions'].values()]

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "orgforg",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons
                #class_names = class_names
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "orgforg":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "orgforg":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
def train(model):
    """Train the model."""
    # Training dataset. 
    dataset_train = orgforgDataset()
    dataset_train.load_orgforg(SAMPLES_DIR, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = orgforgDataset()
    dataset_val.load_orgforg(SAMPLES_DIR, "val")
    dataset_val.prepare()    


    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    #callbacks = [ModelCheckpoint(filepath= logs/, verbose=1,
                #  save_best_only=True, save_weights_only=False)]


    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                #callbacks = callbacks,
                layers='heads')  
    
#     model.save('64x3-CNN.model')

# checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) #



if __name__ == '__main__':
    print('Train')
    
    config = orgforgConfig()
    
    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)
    
    weights_path = COCO_WEIGHTS_PATH
#     COCO_WEIGHTS_PATH = '../../../mask_rcnn_coco.h5'
    
    # Find last trained weights
    # weights_path = model.find_last()[1]
    
    
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    
    train(model)
