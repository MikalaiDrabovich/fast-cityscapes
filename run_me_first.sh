#!/bin/sh
work_dir=$PWD
export CITYSCAPES_DATASET_IN=$work_dir/leftImg8bit_trainvaltest/leftImg8bit/val/
python prepare_results.py


