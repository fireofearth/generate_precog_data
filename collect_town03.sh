#!/bin/bash

SAVEDIR=/home/fireofearth/data/precog_generate/datasets/20210217
python run.py \
    --dir $SAVEDIR \
    --map Town03 \
    --n-episodes 3 \
