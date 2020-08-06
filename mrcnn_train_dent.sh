#!/bin/bash

### 
### Trains the model from folder scratch inside bucket cip.segmentation.
### If need to do another label/folder, change the following line appropriately.

unzip -nq mrcnn.zip
python mrcnn_training_dent.py --dataset=dent/dataset --bucket=cip.segmentation --downloadTrainingData=Y