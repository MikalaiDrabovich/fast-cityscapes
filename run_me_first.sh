#!/bin/sh
work_dir=$PWD

  export CITYSCAPES_DATASET_IN=$work_dir/leftImg8bit_trainvaltest/leftImg8bit/val/
  export CITYSCAPES_DATASET=$work_dir/leftImg8bit_trainvaltest/leftImg8bit/
  export CITYSCAPES_RESULTS=$work_dir/results/
  
  if [ ! -d "$work_dir/cityscapesScripts/" ]; then
    git clone https://github.com/mcordts/cityscapesScripts.git
    cd cityscapesScripts
    git checkout 945c110
    python setup.py build_ext --inplace
  fi

  python prepare_results.py


