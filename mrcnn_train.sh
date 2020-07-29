#!/bin/bash

### 
### Trains the model from folder scratch inside bucket cip.segmentation.
### If need to do another label/folder, change the following line appropriately.

unzip -nq mrcnn.zip
python mrcnn_training.py --dataset=scratch --bucket=cip.segmentation