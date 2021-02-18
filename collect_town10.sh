#!/bin/bash

SAVEDIR=/media/external/data/precog_generate/datasets/20210201
python run.py \
    --dir $SAVEDIR \
    --augment-data \
    --n-vehicles 50 \
    --n-frames 1000 \
    --n-burn-frames 60 \
    --map Town10HD \
