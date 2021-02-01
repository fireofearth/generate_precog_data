#!/bin/bash

SAVEDIR=/media/external/data/precog_generate/datasets/20210127
python run.py \
    --dir $SAVEDIR \
    --augment-data \
    --map Town03 \
