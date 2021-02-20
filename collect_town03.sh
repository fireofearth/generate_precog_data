#!/bin/bash

#SAVEDIR=/media/external/data/precog_generate/datasets/20210211
SAVEDIR=/home/fireofearth/data/precog_generate/datasets/20210217
python run.py \
    --dir $SAVEDIR \
    --map Town03 \
    --n-episodes 3 \
