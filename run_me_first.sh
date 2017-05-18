#!/bin/sh
work_dir=$PWD

export CITYSCAPES_DATASET_IN=$work_dir/leftImg8bit_trainvaltest/leftImg8bit/val/
export CITYSCAPES_DATASET=$work_dir/leftImg8bit_trainvaltest/leftImg8bit/
export CITYSCAPES_RESULTS=$work_dir/results/

python prepare_results.py


