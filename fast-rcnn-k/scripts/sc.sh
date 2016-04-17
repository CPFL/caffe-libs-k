#!/bin/bash

ITERS=120
#ITERS=100
#ITERS=3
#ITERS=40

#NUMS="01 02 04 08 16"
#NUMS="08"
NUMS=""
#NUMS="01 02 03 04 05 06 07 08"

if [ ! -d fast-rcnn-k ]; then
  git clone git@github.com:CPFL/fast-rcnn-k.git
fi

cd fast-rcnn-k/

if [ ! -e Ubuntu14-set ]; then
  if [ ! -d ~/k/Ubuntu14-set ]; then
    tar xzf ~/k/Ubuntu14-set.tgz -C ~/k/
  fi
  ln -s ~/k/Ubuntu14-set
fi

if [ ! -e VOCdevkit ]; then
  if [ ! -d ~/k/VOCdevkit ]; then
    tar xzf ~/k/Ubuntu14-set/VOCdevkit-2007.tgz -C ~/k/
  fi
  ln -s ~/k/VOCdevkit
fi

if [ ! -e intel-doc ]; then
  if [ ! -d ~/k/intel-doc ]; then
    tar xzf ~/k/intel-doc.tgz -C ~/k/
  fi
  ln -s ~/k/intel-doc
fi

PATCH_DIR=patches
BASE_PATCH="verbose.patch"

INTEL_PATCH="intel-1.patch intel-2.patch"
#INTEL_PATCH="intel-1.patch intel-2.patch intel-unroll.patch"
#INTEL_PATCH="intel-1.patch intel-2.patch intel-O3.patch"

#LAYER_OMP_ROI_POOL_PATCH="omp.patch omp_caffe_roi_pool.patch"
LAYER_OMP_ROI_POOL_PATCH="omp.patch omp_caffe_roi_pool.patch omp_caffe_roi_pool-2.patch"
LAYER_OMP_PATCH="$LAYER_OMP_ROI_POOL_PATCH omp_caffe_relu_drop.patch"

TGZ_DIR=intel-doc
BIN_TGZ="$TGZ_DIR/gcc-bin.tgz"

ICC_PATH=/opt/intel/bin/icc
LIBPATH_INTEL="/opt/intel/lib/intel64"
LIBPATH_MKL="/opt/intel/mkl/lib/intel64"

LNKFLG=""

if [ $(hostname) = hpc4.coi.nagoya-u.ac.jp ]; then
  BASE_PATCH="verbose.patch hpc4.patch"
  BIN_TGZ="$TGZ_DIR/gcc-bin-hpc4.tgz"
  LIBPATH_ANACONDA2=/home/axe/k/anaconda2/lib
  LNKFLG="-Wl,-rpath,$LIBPATH_ANACONDA2"
fi

export PYTHONPATH=$(pwd)/fast-rcnn/caffe-fast-rcnn/python:$PYTHONPATH

LEARN_SC=experiments/scripts/default_caffenet.sh
OPT_CPU=cpu
OPT_GPU=0


if [ ! -d fast-rcnn ] && [ -e fast-rcnn-clean.tgz ]; then
  tar xzf fast-rcnn-clean.tgz
fi

if [ ! -e fast-rcnn/data/VOCdevkit2007 ]; then
  (cd fast-rcnn/data/ ; ln -s ../../VOCdevkit VOCdevkit2007 )
fi
if [ ! -e fast-rcnn/data/imagenet_models/CaffeNet.v2.caffemodel ]; then
  tar xzf Ubuntu14-set/imagenet_models.tgz -C fast-rcnn/data/
fi
if [ ! -e fast-rcnn/data/selective_search_data/voc_2007_trainval.mat ]; then
  tar xzf Ubuntu14-set/selective_search_data.tgz -C fast-rcnn/data/
fi

if [ ! -e fast-rcnn/caffe-fast-rcnn/Makefile.config ]; then
  cp Ubuntu14-set/Makefile.config fast-rcnn/caffe-fast-rcnn/
fi

if [ ! -e fast-rcnn-clean.tgz ]; then
  tar czf fast-rcnn-clean.tgz fast-rcnn
fi

if [ ! -e ~/matlab ]; then
  touch ~/matlab
  chmod a+x ~/matlab
fi

if [ ! -d logs ]; then
  mkdir logs
fi

if [ ! -d fast-rcnn-baks ]; then
  mkdir fast-rcnn-baks
fi

#
# OpenBLAS package
#
# need install
#
# $ sudo apt-get install libopenblass-base
# $ sudo apt-get install libopenblass-dev
#

#
# GPU libcudnn
#
# need install
#
# $ tar xzf intel-doc/cudnn-6.5-linux-x64-v2.tgz
# $ (cd cudnn-6.5-linux-x64-v2/ ; tar cf - libcudnn* ) | sudo tar xf - -C /usr/local/cuda/lib64
# $ sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include/
# $ sudo ldconfg
#


#
# OpenBLAS build
#

BLAS_SRC_DIR=xianyi-OpenBLAS-53e849f

if [ ! -e OpenBLAS-clean.tgz ]; then
  unzip intel-doc/xianyi-OpenBLAS-v0.2.15-0-g53e849f.zip
  tar czf OpenBLAS-clean.tgz $BLAS_SRC_DIR
fi

NAME=gcc_open_blas
INSTALL_DIR=$(pwd)/$NAME
if [ ! -d $INSTALL_DIR ]; then
  echo "@ $NAME (OpenBLAS GCC build)"

  if [ ! -d $BLAS_SRC_DIR ]; then
    tar xzf OpenBLAS-clean.tgz $BLAS_SRC_DIR

    PATCHS="gcc_open.patch"
    if [ $(hostname) = hpc4.coi.nagoya-u.ac.jp ]; then
      PATCHS="$PATCHS open_hpc4.patch"
    fi
    (cd $PATCH_DIR ; cat $PATCHS ) | (cd $BLAS_SRC_DIR/ ; patch -p1 )
  fi

  (cd $BLAS_SRC_DIR/ ; make USE_OPENMP=1 )

  tar czf blas-gcc-bin.tgz -C $BLAS_SRC_DIR interface/dscal.o kernel/dscal_k.o kernel/cscal_k.o interface/zscal.o kernel/zscal_k.o

  (cd $BLAS_SRC_DIR/ ; make install PREFIX=$INSTALL_DIR )

  rm -rf $BLAS_SRC_DIR-$NAME
  mv $BLAS_SRC_DIR $BLAS_SRC_DIR-$NAME
fi
INSTALL_DIR=""

NAME=intel_open_blas
INSTALL_DIR=$(pwd)/$NAME
if [ ! -d $INSTALL_DIR -a -e $ICC_PATH ]; then
  echo "@ $NAME (OpenBLAS Intel build)"

  if [ ! -d $BLAS_SRC_DIR ]; then
    tar xzf OpenBLAS-clean.tgz $BLAS_SRC_DIR

    PATCHS="intel_open.patch"
    if [ $(hostname) = hpc4.coi.nagoya-u.ac.jp ]; then
      PATCHS="$PATCHS intel_open_hpc4.patch open_hpc4.patch"
    fi
    (cd $PATCH_DIR ; cat $PATCHS ) | (cd $BLAS_SRC_DIR/ ; patch -p1 )

    tar xzf blas-gcc-bin.tgz -C $BLAS_SRC_DIR/
    tar tzf blas-gcc-bin.tgz | (cd $BLAS_SRC_DIR/ ; xargs touch )
  fi

  (cd $BLAS_SRC_DIR/ ; make CC=$ICC_PATH USE_OPENMP=1 )
  (cd $BLAS_SRC_DIR/ ; make install PREFIX=$INSTALL_DIR )

  rm -rf $BLAS_SRC_DIR-$NAME
  mv $BLAS_SRC_DIR $BLAS_SRC_DIR-$NAME
fi
INSTALL_DIR=""


# caffe

name_chk() {
  echo $NAME | grep $1 > /dev/null
  echo ";;; name_chk NAME=$NAME pat=$1 ret=$?"

  echo $NAME | grep $1 > /dev/null
}

ins_path() {
  OLD=$(eval echo '$'"$1")
  INS="$2"
  if [ x"$OLD" != x ]; then
    if [ x"$3" != x ]; then
      INS="$INS$3"
    else
      INS="$INS:"
    fi
  fi
  eval "$1=\"$INS$OLD\""
}

modify_makefile_config() {
  echo ";;; modify Makefile.config"

  LFLG="$LNKFLG"

  if name_chk "-e intel_caffe -e intel_omp_caffe"; then
    ins_path LFLG "-Wl,-rpath,$LIBPATH_INTEL" ' '
  fi

  ADD=""
  if name_chk gcc_open_blas; then
    ins_path LFLG "-Wl,-rpath,$(pwd)/gcc_open_blas/lib" ' '
  fi
  if name_chk intel_open_blas; then
    ins_path LFLG "-Wl,-rpath,$(pwd)/intel_open_blas/lib -Wl,-rpath,$LIBPATH_INTEL" ' '
  fi
  if name_chk mkl_blas; then
    ins_path LFLG "-Wl,-rpath,$LIBPATH_MKL -Wl,-rpath,$LIBPATH_INTEL" ' '
  fi

  echo "LINKFLAGS += $LFLG" >> fast-rcnn/caffe-fast-rcnn/Makefile.config

  tail -1 fast-rcnn/caffe-fast-rcnn/Makefile.config
}

apply_patchs () {
  if [ $# -gt 1 ]; then
    PTS="$1"
  else
    PTS="$BASE_PATCH"
    if name_chk _omp_; then
      if name_chk _omp_caffe_roi_pool_; then
        PTS="$PTS $LAYER_OMP_ROI_POOL_PATCH layer_time.patch"
        #PTS="$PTS $LAYER_OMP_ROI_POOL_PATCH layer_time.patch cache-test.patch"
      else
        PTS="$PTS $LAYER_OMP_PATCH layer_time.patch"
        #PTS="$PTS $LAYER_OMP_PATCH layer_time.patch cache-test.patch"
      fi
    fi
    if name_chk _fmax; then
      PTS="$PTS fmax.patch"
    fi
    if name_chk _layer_time; then
      PTS="$PTS layer_time.patch"
    fi
    if name_chk "-e intel_caffe -e intel_omp_caffe"; then
      PTS="$PTS $INTEL_PATCH"
    fi
    if name_chk _open_blas; then
      PTS="$PTS open.patch"
    fi
    if name_chk mkl_blas; then
      PTS="$PTS mkl.patch"
    fi
    if name_chk gpu; then
      PTS="$PTS gpu.patch"
    fi
  fi

  echo ";;; apply patchs $PTS"

  (cd $PATCH_DIR ; cat $PTS ) | (cd fast-rcnn ; patch -p1 )

  modify_makefile_config
}

copy_gcc_bin() {
  if [ x"$BIN_TGZ" != x ]; then
    echo ";;; $BIN_TGZ"
    tar xzf $BIN_TGZ && (tar tzf $BIN_TGZ | xargs touch )
  fi

  SRC=fast-rcnn-baks/fast-rcnn-$(echo $NAME | sed s/^intel/gcc/)/caffe-fast-rcnn/python/caffe/_caffe.so
  if [ -e "$SRC" ]; then
    echo ";;; copy $SRC"
    cp "$SRC" fast-rcnn/caffe-fast-rcnn/python/caffe/
  fi
}

build () {
  if name_chk "-e intel_caffe -e intel_omp_caffe"; then
    copy_gcc_bin
  fi

  MKOPT=""
  if name_chk _omp_; then
    MKOPT="OPENMP=1"
  fi

  (cd fast-rcnn/lib/ ; make )

  echo ";;; MKOTP=$MKOPT"

  (cd fast-rcnn/caffe-fast-rcnn/ ; make $MKOPT ; make $MKOPT pycaffe )
}

setup() {
  if [ $1 = before ]; then
    LD_PATH=""
    if name_chk mkl_blas; then
      ins_path LD_PATH $LIBPATH_INTEL
      ins_path LD_PATH $LIBPATH_MKL
    fi
    export LD_LIBRARY_PATH="$LD_PATH"
    echo ";;; LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

    if name_chk darshan; then
      if [ ! -d ~/k/darshan-aics ]; then
        tar xzf $TGZ_DIR/darshan-aics-source-20160223.tar.gz -C ~/k
        cat $PATCH_DIR/darshan_exit.patch | (cd ~/k/darshan-aics ; patch -p1 )
        (cd ~/k/darshan-aics/darshan-2.3.0/darshan-runtime/ ; 
         ./configure --with-mem-align=8 --with-log-path=./darshan --with-jobid-env=PBS_JOBID CC=mpicc CFLAGS="-O2 -g -DHISTORY" ;
         make )
      fi
    fi
  fi # before

  if [ $1 = after ]; then
    echo ";;; after ..."
  fi # after
}

backup_clean() {

  rm -rf fast-rcnn-baks/fast-rcnn-$NAME
  echo ";;; backup fast-rcnn --> fast-rcnn-$NAME"
  mv fast-rcnn fast-rcnn-baks/fast-rcnn-$NAME

  tar xzf fast-rcnn-clean.tgz
  echo ";;; prepared fast-rcnn clean"
}

run_learn() {
  setup before

  OPT_CPU_GPU=$OPT_CPU
  if name_chk gpu; then
    OPT_CPU_GPU=$OPT_GPU
  fi

  if name_chk darshan; then
    (cd fast-rcnn/ ; \
     export LD_PRELOAD=~/k/darshan-aics/darshan-2.3.0/darshan-runtime/lib/libdarshan-single.so; \
     export DARSHAN_HISTORY_RW="rw"; \
     PATH=$PATH:~ $LEARN_SC $OPT_CPU_GPU --iters $ITERS ) |& tee logs/$NAME.log
  else
    (cd fast-rcnn/ ; PATH=$PATH:~ $LEARN_SC $OPT_CPU_GPU --iters $ITERS ) | tee logs/$NAME.log
  fi

  if [ "$?" != "0" -a $OPT_CPU_GPU = $OPT_CPU ]; then
    for N in $NUMS; do
      THS=$(echo $N | sed 's/^0*//')
      (cd fast-rcnn/ ; PATH=$PATH:~ OMP_NUM_THREADS=$THS $LEARN_SC $OPT_CPU_GPU --iters $ITERS ) | tee logs/$NAME-$N.log
    done
  fi

  setup after

  backup_clean
}

patch_build_run() {
  apply_patchs "$*"
  build

  if [ $NAME = "gcc_caffe_atlas_blas" ]; then
    if [ ! -e $BIN_TGZ ]; then
      tar czf $BIN_TGZ \
        fast-rcnn/caffe-fast-rcnn/.build_release/src/caffe/layer_factory.o
      echo ";;; gcc bin --> $BIN_TGZ"
    fi
  fi

  run_learn
}

#
# gcc_caffe
#

NAME=gcc_caffe_atlas_blas
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC Caffe + ATLAS BLAS)"
  patch_build_run
fi

NAME=gcc_caffe_open_blas
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC Caffe + OpenBLAS package)"
  patch_build_run
fi

#NAME=gcc_caffe_gcc_open_blas
#if [ ! -e logs/$NAME.log ]; then
#  echo "@ $NAME (GCC Caffe + GCC build OpenBLAS)"
#  patch_build_run
#fi

NAME=gcc_caffe_intel_open_blas
if [ ! -e logs/$NAME.log -a -d $(pwd)/intel_open_blas/lib ]; then
  echo "@ $NAME (GCC Caffe + Intel build OpenBLAS)"
  patch_build_run
fi

NAME=gcc_caffe_mkl_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (GCC Caffe + Intel MKL BLAS)"
  patch_build_run
fi

NAME=gcc_caffe_gpu
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC Caffe + GPU)"
  patch_build_run
fi


#
# intel_caffe
#

NAME=intel_caffe_atlas_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build Caffe + ATLAS BLAS)"
  patch_build_run
fi

NAME=intel_caffe_open_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build Caffe + OpenBLAS Package)"
  patch_build_run
fi

NAME=intel_caffe_intel_open_blas
if [ ! -e logs/$NAME.log -a -d $(pwd)/intel_open_blas/lib ]; then
  echo "@ $NAME (Intel build Caffe + Intel build OpenBLAS)"
  patch_build_run
fi

NAME=intel_caffe_mkl_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build Caffe + Intel MKL BLAS)"
  patch_build_run
fi

#NAME=intel_caffe_gpu
#if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
#  echo "@ $NAME (Intel build Caffe + GPU)"
#  patch_build_run
#fi


#
# BLAS time
#

#NAME=gcc_caffe_atlas_blas_time
#if [ ! -e logs/$NAME.log ]; then
#  echo "@ $NAME (GCC Caffe + ATLAS BLAS , time)"
#  patch_build_run "$BASE_PATCH blas_time.patch"
#fi

#NAME=intel_caffe_mkl_blas_time
#if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
#  echo "@ $NAME (Intel build Caffe + Intel MKL BLAS, time)"
#  patch_build_run "$BASE_PATCH $INTEL_PATCH mkl.patch blas_time.patch"
#fi


#
# Layer time
#

NAME=gcc_caffe_atlas_blas_layer_time
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC Caffe + ATLAS BLAS , Layer time)"
  patch_build_run
fi

NAME=gcc_caffe_open_blas_layer_time
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC Caffe + OpenBLAS package , Layer time)"
  patch_build_run
fi

NAME=gcc_caffe_intel_open_blas_layer_time
if [ ! -e logs/$NAME.log -a -d $(pwd)/intel_open_blas/lib ]; then
  echo "@ $NAME (GCC Caffe + Intel build OpenBLAS , Layer time)"
  patch_build_run
fi

NAME=gcc_caffe_mkl_blas_layer_time
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (GCC Caffe + Intel MKL BLAS , Layer time)"
  patch_build_run
fi

NAME=intel_caffe_atlas_blas_layer_time
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build Caffe + ATLAS BLAS , Layer time)"
  patch_build_run
fi

NAME=intel_caffe_open_blas_layer_time
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build Caffe + OpenBLAS Package , Layer time)"
  patch_build_run
fi

NAME=intel_caffe_intel_open_blas_layer_time
if [ ! -e logs/$NAME.log -a -d $(pwd)/intel_open_blas/lib ]; then
  echo "@ $NAME (Intel build Caffe + Intel build OpenBLAS , Layer time)"
  patch_build_run
fi

NAME=intel_caffe_mkl_blas_layer_time
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build Caffe + Intel MKL BLAS, Layer time)"
  patch_build_run "$BASE_PATCH $INTEL_PATCH layer_time.patch mkl.patch"
fi

NAME=gcc_caffe_gpu_layer_time
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC Caffe + GPU, Layer time)"
  patch_build_run "$BASE_PATCH gpu.patch layer_time.patch"
fi


#
# gcc_omp_caffe
#

NAME=gcc_omp_caffe_atlas_blas
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC OMP Caffe + ATLAS BLAS)"
  patch_build_run
fi

NAME=gcc_omp_caffe_open_blas
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC OMP Caffe + OpenBLAS package)"
  patch_build_run
fi

NAME=gcc_omp_caffe_intel_open_blas
if [ ! -e logs/$NAME.log -a -d $(pwd)/intel_open_blas/lib ]; then
  echo "@ $NAME (GCC OMP Caffe + Intel build OpenBLAS)"
  patch_build_run
fi

NAME=gcc_omp_caffe_mkl_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (GCC OMP Caffe + Intel MKL BLAS)"
  patch_build_run
fi


#
# intel_omp_caffe
#

NAME=intel_omp_caffe_atlas_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build OMP Caffe + ATLAS BLAS)"
  patch_build_run
fi

NAME=intel_omp_caffe_open_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build OMP Caffe + OpenBLAS Package)"
  patch_build_run
fi

NAME=intel_omp_caffe_intel_open_blas
if [ ! -e logs/$NAME.log -a -d $(pwd)/intel_open_blas/lib ]; then
  echo "@ $NAME (Intel build OMP Caffe + Intel build OpenBLAS)"
  patch_build_run
fi

NAME=intel_omp_caffe_mkl_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build OMP Caffe + Intel MKL BLAS)"
  patch_build_run "$BASE_PATCH $INTEL_PATCH $LAYER_OMP_PATCH layer_time.patch mkl.patch"
fi

NAME=intel_omp_caffe_mkl_blas_fmax
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build OMP Caffe + Intel MKL BLAS + fmax)"
  patch_build_run "$BASE_PATCH $INTEL_PATCH $LAYER_OMP_PATCH fmax.patch layer_time.patch mkl.patch"
fi


#
# omp roi_pool only
#

NAME=gcc_omp_caffe_roi_pool_atlas_blas
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC OMP roi_pool Caffe + ATLAS BLAS)"
  patch_build_run
fi

NAME=gcc_omp_caffe_roi_pool_open_blas
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC OMP roi_pool Caffe + OpenBLAS package)"
  patch_build_run
fi

NAME=gcc_omp_caffe_roi_pool_intel_open_blas
if [ ! -e logs/$NAME.log -a -d $(pwd)/intel_open_blas/lib ]; then
  echo "@ $NAME (GCC OMP roi_pool Caffe + Intel build OpenBLAS)"
  patch_build_run
fi

NAME=gcc_omp_caffe_roi_pool_mkl_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (GCC OMP roi_pool Caffe + Intel MKL BLAS)"
  patch_build_run
fi

NAME=intel_omp_caffe_roi_pool_atlas_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build OMP roi_pool Caffe + ATLAS BLAS)"
  patch_build_run
fi

NAME=intel_omp_caffe_roi_pool_open_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build OMP roi_pool Caffe + OpenBLAS Package)"
  patch_build_run
fi

NAME=intel_omp_caffe_roi_pool_intel_open_blas
if [ ! -e logs/$NAME.log -a -d $(pwd)/intel_open_blas/lib ]; then
  echo "@ $NAME (Intel build OMP roi_pool Caffe + Intel build OpenBLAS)"
  patch_build_run
fi

NAME=intel_omp_caffe_roi_pool_mkl_blas
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build OMP roi_pool Caffe + Intel MKL BLAS)"
  patch_build_run "$BASE_PATCH $INTEL_PATCH $LAYER_OMP_ROI_POOL_PATCH layer_time.patch mkl.patch"
fi

NAME=intel_omp_caffe_roi_pool_mkl_blas_fmax
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build OMP roi_pool Caffe + Intel MKL BLAS + fmax)"
  patch_build_run "$BASE_PATCH $INTEL_PATCH $LAYER_OMP_ROI_POOL_PATCH fmax.patch layer_time.patch mkl.patch"
fi

#
# darshan
#

NAME=gcc_caffe_atlas_blas_darshan
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC Caffe + ATLAS BLAS + darshan)"
  patch_build_run
fi

NAME=intel_caffe_mkl_blas_darshan
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build Caffe + Intel MKL BLAS + darshan)"
  patch_build_run
fi

NAME=gcc_omp_caffe_roi_pool_atlas_blas_darshan
if [ ! -e logs/$NAME.log ]; then
  echo "@ $NAME (GCC OMP roi_pool Caffe + ATLAS BLAS + darshan)"
  patch_build_run
fi

NAME=intel_omp_caffe_roi_pool_mkl_blas_darshan
if [ ! -e logs/$NAME.log -a -e $ICC_PATH ]; then
  echo "@ $NAME (Intel build OMP roi_pool Caffe + Intel MKL BLAS + darshan)"
  patch_build_run "$BASE_PATCH $INTEL_PATCH $LAYER_OMP_ROI_POOL_PATCH layer_time.patch mkl.patch darshan.patch"
fi


#
# logs
#

if [ ! -e time_tool ]; then
  gcc -o time_tool intel-doc/time_tool.c
fi

if [ ! -e log_fmt ]; then
  gcc -o log_fmt intel-doc/log_fmt.c
fi

LST="\
gcc_caffe_atlas_blas \
gcc_caffe_atlas_blas_layer_time \
gcc_caffe_open_blas \
gcc_caffe_open_blas_layer_time \
gcc_caffe_intel_open_blas \
gcc_caffe_intel_open_blas_layer_time \
gcc_caffe_mkl_blas \
gcc_caffe_mkl_blas_layer_time \
gcc_caffe_gpu \
gcc_caffe_gpu_layer_time \
intel_caffe_atlas_blas \
intel_caffe_atlas_blas_layer_time \
intel_caffe_open_blas \
intel_caffe_open_blas_layer_time \
intel_caffe_intel_open_blas \
intel_caffe_intel_open_blas_layer_time \
intel_caffe_mkl_blas \
intel_caffe_mkl_blas_layer_time \
gcc_omp_caffe_atlas_blas \
gcc_omp_caffe_open_blas \
gcc_omp_caffe_intel_open_blas \
gcc_omp_caffe_mkl_blas \
intel_omp_caffe_atlas_blas \
intel_omp_caffe_open_blas \
intel_omp_caffe_intel_open_blas \
intel_omp_caffe_mkl_blas \
intel_omp_caffe_mkl_blas_fmax \
gcc_omp_caffe_roi_pool_atlas_blas \
gcc_omp_caffe_roi_pool_open_blas \
gcc_omp_caffe_roi_pool_intel_open_blas \
gcc_omp_caffe_roi_pool_mkl_blas \
intel_omp_caffe_roi_pool_atlas_blas \
intel_omp_caffe_roi_pool_open_blas \
intel_omp_caffe_roi_pool_intel_open_blas \
intel_omp_caffe_roi_pool_mkl_blas \
intel_omp_caffe_roi_pool_mkl_blas_fmax"

log_sub () {
  FN=$1
  if [ -e $FN ]; then
    echo $FN

    START=$(sed -n 's/\..* Iteration 0, loss.*$//p' $FN | sed 's/^.* //')
    STOP=$(sed -n 's/\..* Iteration 100, loss.*$//p' $FN | sed 's/^.* //')
    ../time_tool $STOP $START 

    grep 'BLAS sec' $FN | tail -1 | sed 's/^.*\] //'
    grep ' fwd=' $FN | sed 's/^.*\] //'
    echo ""
  fi
}

(cd logs
  for NM in $LST; do
    log_sub $NM.log
  done

  for N in $NUMS; do
    for NM in $LST; do
      log_sub $NM-$N.log
    done
  done
) |& tee logs/time.txt | ./log_fmt > logs/time_fmt.txt

cp ~/k/intel-doc/sc.sh logs/
tar czf logs-$(date +%y%m%d-%H).tgz logs

# EOF
