#!/bin/sh
work_dir=$PWD
if [ -d "$work_dir/leftImg8bit_trainvaltest/leftImg8bit/val/" ]; then
  export CITYSCAPES_DATASET_IN=$work_dir/leftImg8bit_trainvaltest/leftImg8bit/val/
  export CITYSCAPES_DATASET=$work_dir/leftImg8bit_trainvaltest/leftImg8bit/
  export CITYSCAPES_RESULTS=$work_dir/results/
  
  if [ ! -d "$work_dir/results/" ]; then
    mkdir results
  fi
  
  if [ ! -d "$work_dir/cityscapesScripts/" ]; then
    git clone https://github.com/mcordts/cityscapesScripts.git
    cd cityscapesScripts
    git checkout 945c110
    python setup.py build_ext --inplace
  fi

  python prepare_results.py
    
  cd $work_dir/cityscapesScripts/cityscapesscripts/evaluation
  python evalPixelLevelSemanticLabeling.py
    
else
  echo 'You need to download leftImg8bit_trainvaltest.zip (11GB) from https://www.cityscapes-dataset.com/downloads and unpack it to current directory' $work_dir
  echo 'Then you need to download gtFine_trainvaltest.zip (241MB) and unpack it to '$work_dir'/leftImg8bit_trainvaltest/leftImg8bit/'
  echo 'Finally, copy your trained models into ./models/ or download and unpack them from https://github.com/MikalaiDrabovich/fast-pixel-classification-dnn-caffe/archive/v0.1.zip'
fi


