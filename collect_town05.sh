#!/bin/bash

python run.py \
    --dir /media/external/data/precog_generated_dataset/town05/unsorted \
    --augment-data \
    --n-vehicles 100 \
    --n-frames 1500 \
    --n-burn-frames 500 \
    --map Town05 \