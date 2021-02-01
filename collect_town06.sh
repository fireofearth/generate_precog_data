#!/bin/bash

# twice as many episodes for half the time
SAVEDIR=/media/external/data/precog_generate/datasets/20210127
python run.py \
    --dir $SAVEDIR \
    --augment-data \
    --n-vehicles 130 \
    --n-episodes 20 \
    --n-frames 560 \
    --n-burn-frames 60 \
    --map Town06 \
