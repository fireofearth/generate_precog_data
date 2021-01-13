#!/bin/bash

# twice as many episodes for half the time
python run.py \
    --dir /media/external/data/precog_generated_dataset/town06/unsorted \
    --augment-data \
    --n-vehicles 130 \
    --n-episodes 20 \
    --n-frames 560 \
    --n-burn-frames 60 \
    --map Town06 \
