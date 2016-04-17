#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/default_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ $# -ge 1 ]; then
  if [ $1 = "cpu" ]; then
    GPU="--cpu"
  else
    GPU="--gpu $1"
  fi
  shift
fi

if [ $# -ge 2 -a $1 = "--iters" ]; then
  ITERS="$1 $2"
  shift
  shift
fi

time ./tools/train_net.py $GPU $ITERS \
  --solver models/CaffeNet/solver.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb voc_2007_trainval

#time ./tools/test_net.py $GPU \
#  --def models/CaffeNet/test.prototxt \
#  --net output/default/voc_2007_trainval/caffenet_fast_rcnn_iter_40000.caffemodel \
#  --imdb voc_2007_test
