#!/bin/bash

SAVEDIR=/media/external/data/precog_generate/datasets/20210211
python run.py \
    --dir $SAVEDIR \
    --map Town03 \
    --n-episodes 66 \
