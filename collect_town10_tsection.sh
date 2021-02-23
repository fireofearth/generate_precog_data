#!/bin/bash

SAVEDIR=/home/fireofearth/data/precog_generate/datasets/20210217/30vehicles
python run_town10_tsection.py \
    --dir $SAVEDIR \
    --n-episodes 300 \
    --n-vehicles 30
