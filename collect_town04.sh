#!/bin/bash

python run.py \
    --dir /media/external/data/precog_generated_dataset/town04/unsorted \
    --augment-data \
    --n-vehicles 120 \
    --n-frames 2000 \
    --n-burn-frames 1000 \
    --map Town04 \
