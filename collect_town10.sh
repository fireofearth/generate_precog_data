#!/bin/bash

python run.py \
    --dir /media/external/data/precog_generated_dataset/town10/unsorted \
    --augment-data \
    --n-vehicles 80 \
    --n-frames 1000 \
    --n-burn-frames 60 \
    --map Town10HD \
    -v \
