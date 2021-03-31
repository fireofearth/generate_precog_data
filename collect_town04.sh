#!/bin/bash

SAVEDIR=/home/fireofearth/data/precog_generate/datasets/20210329
    # --augment-data \
python run.py \
    --dir $SAVEDIR \
    --n-vehicles 120 \
    --n-frames 1500 \
    --n-burn-frames 500 \
    --map Town04 \
