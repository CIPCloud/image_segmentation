"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""
import boto3
import json
import os
import sys

import numpy as np
import skimage.draw
import pathlib
import re
# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_DIR = os.path.join(ROOT_DIR, "model")
COCO_WEIGHTS_PATH = os.path.join(COCO_MODEL_DIR,"mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
EPOCHS=30
MODEL_BUCKET="cip.models"
DIR_PATTERN = re.compile(".*/$")
MODEL_TYPE="segmentation"
TRAINING_LABEL="scratch"

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = TRAINING_LABEL

    # We use a GPU with 6GB memory, which can fit only one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Car Background + scratch

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class(TRAINING_LABEL, 1, TRAINING_LABEL)

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
        annotations1 = json.load(
            open(os.path.join(dataset_dir, "via_region_data.json"), 'r', encoding="utf8", errors='ignore'))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                TRAINING_LABEL,  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != TRAINING_LABEL:
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
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == TRAINING_LABEL:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName) 
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        print(">>Processing:{}".format(obj.key))
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        if DIR_PATTERN.match(obj.key):
            continue
        bucket.download_file(obj.key, obj.key)  
        
def uploadModel(bucket, model):
    model_path = model.find_last()
    model_file= os.path.basename(model_path)
    model_dir = os.path.join(MODEL_TYPE,TRAINING_LABEL,pathlib.PurePath(model_path).parent.name)
    print("Model path:{} model_dir={} model:{}".format(model_path,model_dir,model_file))
    
    s3 = boto3.client('s3')
    try:
        response = s3.put_object(Bucket=bucket, Key=model_dir +'/')
        print("  %s", response)
        s3.upload_file(model_path,MODEL_BUCKET, os.path.join(model_dir,model_file))
    except Exception as e:
        print("Bucket error %s", e)
    
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # Download mask_rcnn_coco.h5 weights before starting the training
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=EPOCHS,
                layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse
    if not os.path.exists(COCO_MODEL_DIR):
        os.makedirs(COCO_MODEL_DIR)
    if not os.path.exists(DEFAULT_LOGS_DIR):
        os.makedirs(DEFAULT_LOGS_DIR)
    # Parse command line arguments
    parser = argparse.ArgumentParser(  description='Train Mask R-CNN to detect custom class.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the training dataset')
    parser.add_argument('--bucket', required=False,
                        metavar="bucket name",
                        help='name of s3 bucket')
    parser.add_argument('--downloadTrainingData', required=False,
                        metavar="Y/N",
                        help='should we download from s3 bucket')
    args = parser.parse_args()

    if (args.downloadTrainingData=="Y") and (args.bucket is not None):
        print("Downloading from bucket:{} dir:{}".format(args.bucket,args.dataset))
        if os.path.exists(args.dataset):
            rename_dir=args.dataset+"__tmp"
            print(">>Renaming existing dir before fetching from s3:{}".format(rename_dir))
            os.rename(args.dataset,rename_dir )
        downloadDirectoryFroms3(args.bucket,args.dataset)
    else:
        print("local training data already available")
        
    # Validate arguments
    assert args.dataset, "Argument --dataset is required for training"
    print("Dataset used for training:{}".format(args.dataset))
    
    # Configurations
    config = CustomConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,   model_dir=DEFAULT_LOGS_DIR)
    # Select weights file to load
    weights_path = COCO_WEIGHTS_PATH
        # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
        

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    
    train(model)
    uploadModel(MODEL_BUCKET,model)