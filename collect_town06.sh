#!/bin/bash

# twice as many episodes for half the time
SAVEDIR=/home/fireofearth/data/precog_generate/datasets/20210329
    # --augment-data \
python run.py \
    --dir $SAVEDIR \
    --n-vehicles 130 \
    --n-episodes 20 \
    --n-frames 560 \
    --n-burn-frames 60 \
    --map Town06 \
