#!/bin/bash

SAVEDIR=/media/external/data/precog_generate/datasets/20210201
python run.py \
    --dir $SAVEDIR \
    --augment-data \
    --map Town03 \
