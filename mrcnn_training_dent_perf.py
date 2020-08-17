import os
import json
import numpy as np
import datetime

from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap, compute_recall
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn import visualize

import matplotlib.pyplot as plt
import skimage.draw
from skimage import color
from skimage import io

class PredictionConfig(Config):
    NAME = 'dent'
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def evaluate_model(dataset, model, cfg):
    ARs = list()
    APs = list() 
    F1_scores = list()
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        AR, positive_ids = compute_recall(r["rois"], gt_bbox, iou=0.2)
        ARs.append(AR)
        F1_scores.append((2* (mean(precisions) * mean(recalls)))/(mean(precisions) + mean(recalls)))
        APs.append(AP)

    mAP = mean(APs)
    mAR = mean(ARs)
        
    return mAP, mAR, F1_scores

############################################################
#  Dataset
############################################################
from mrcnn import utils

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


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    print("Shapes: mask={} image={} gray={}".format(mask.shape, image.shape, gray.shape))
    if image.shape[-1] == 4:
        image = image[..., :3]
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def detect_and_color_splash(model, image_path=None, out_mask_folder=None, out_bb_folder=None):
    assert image_path

    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    print("Detecting images:{}".format(r))
    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    file_name_mask = file_name
    if out_mask_folder is not None:
        file_name_mask = os.path.join(out_mask_folder, file_name_mask)

    skimage.io.imsave(file_name_mask, splash)

    # Create & save pic with bounding boxes
    ax = get_ax(1)
    file_name_bb = file_name.replace('mask_', 'bb_', 1)
    if out_bb_folder is not None:
        file_name_bb = os.path.join(out_bb_folder, file_name_bb)      

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
            ['BG','scratch', ], r['scores'], ax=ax,
            title="scratch instances for:"+ file_name,
            filename=file_name_bb)

    print("Mask file saved to ", file_name_mask)
    print("BB file saved to ", file_name_bb)
    return file_name_mask, file_name_bb, r

if __name__ == '__main__':
	dataset = 'dent-images'
	TRAINING_LABEL = 'dent'

	dataset_val = CustomDataset()
	dataset_val.load_custom(dataset, "val")
	dataset_val.prepare()

	cfg = PredictionConfig()
	model = MaskRCNN(mode='inference', model_dir='/home/ec2-user/source/image_segmentation/model', config=cfg)
	model.load_weights('/home/ec2-user/source/image_segmentation/model/mask_rcnn_dent_0030.h5', by_name=True)
	mAP, mAR, F1_score = evaluate_model(dataset_val, model, cfg)

	print("mAP: %.3f" % mAP)
	print("mAR: %.3f" % mAR)
	print("first way calculate f1-score: ", F1_score)

	F1_score_2 = (2 * mAP * mAR)/(mAP + mAR)
	print('second way calculate f1-score_2: ', F1_score_2)

	imagedir = '/home/ec2-user/source/image_segmentation/dent-images/val'
	imagefiles = [f for f in os.listdir(imagedir) if (os.path.isfile(os.path.join(imagedir, f)) and f.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png'])]

	out_mask_folder = '/home/ec2-user/source/image_segmentation/out-images/mask'
	out_bb_folder = '/home/ec2-user/source/image_segmentation/out-images/bb'
	for f in imagefiles:
	    image_path = os.path.join(imagedir, f)
	    print('Processing image:', f)
	    detect_and_color_splash(model, image_path=image_path, out_mask_folder=out_mask_folder, out_bb_folder=out_bb_folder)

	print("Done...")