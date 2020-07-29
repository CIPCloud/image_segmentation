#!/bin/bash

unzip -nq mrcnn.zip
python mrcnn_training.py "$@"