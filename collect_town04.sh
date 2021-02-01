#!/bin/bash

SAVEDIR=/media/external/data/precog_generate/datasets/20210127
    # --dir $SAVEDIR \
python run.py \
    --augment-data \
    --n-vehicles 120 \
    --n-frames 1500 \
    --n-burn-frames 500 \
    --map Town04 \
