# fast-rcnn-k

Caffe学習動作のインストール、実行手順

2015/DEC/02

------------------------------

OS : Ubuntu 14.04.1 LTS

ソースコード、データセットのアーカイブ一式

Ubuntu14-set/

  オリジナルのソースコード
  fast-rcnn-caffe-src.tgz

  ソースコードへのパッチ
  layers.patch
  script.patch

  Caffeコンフィグ設定
  Makefile.config

  追加Pythonライブラリ
  easydict-1.6.zip (10,282バイト)

  画像データ
  VOCdevkit-2007.tgz (908,144,154バイト)

  サイトからDownloadしたデータ
  imagenet_models.tgz (226,089,730バイト)
  selective_search_data.tgz (37,898,576バイト)

  サイトからDownloadした40,000回学習済のモデルファイル
  caffenet_fast_rcnn_iter_40000.caffemodel (229,195,587バイト)

------------------------------

## 1 インストール手順

```
$ sudo apt-get update
$ sudo apt-get install openssh-server
```

 別端末からlogin
```
$ ssh username@hostname
```

### 1.1 ライブラリのインストール

```
$ sudo apt-get install libnlopt-dev freeglut3-dev qtbase5-dev libqt5opengl5-dev libssh2-1-dev libarmadillo-dev libpcap-dev
$ sudo apt-get install libboost-all-dev libhdf5-dev libopencv-dev
$ sudo apt-get install protobuf-compiler libgoogle-glog-dev libleveldb-dev liblmdb-dev libsnappy-dev libatlas-base-dev
$ sudo apt-get install python-skimage python-protobuf Cython python-pip python-opencv
```


### 1.2 作業ディレクトリの準備

```
$ mkdir work
$ cd work
$ ln -s /somewhere/Ubuntu14-set Ubuntu14-set
```


### 1.3 オリジナルのソースコードを展開

```
$ tar xzf Ubuntu14-set/fast-rcnn-caffe-src.tgz 
```


### 1.4 Pythonライブラリのインストール

```
$ sudo pip install -r fast-rcnn/caffe-fast-rcnn/python/requirements.txt

$ unzip Ubuntu14-set/easydict-1.6.zip 
$ (cd easydict-1.6/ ; sudo python setup.py install )
```


### 1.5 画像データを配置

```
$ tar xzf Ubuntu14-set/VOCdevkit-2007.tgz
$ (cd fast-rcnn/data/ ; ln -s ../../VOCdevkit VOCdevkit2007 )
```


### 1.6 サイトからDownloadしたデータを配置

```
$ tar xzf Ubuntu14-set/imagenet_models.tgz -C fast-rcnn/data/
$ tar xzf Ubuntu14-set/selective_search_data.tgz -C fast-rcnn/data/
```


### 1.7 ソースコードへのパッチ

```
$ cat Ubuntu14-set/layers.patch | patch -p1
$ cat Ubuntu14-set/script.patch | patch -p1
```

------------------------------

## 2 ビルド手順

```
$ (cd fast-rcnn/lib/ ; make )

$ cp Ubuntu14-set/Makefile.config fast-rcnn/caffe-fast-rcnn/Makefile.config
$ (cd fast-rcnn/caffe-fast-rcnn/ ; make ; make pycaffe )
```

------------------------------

## 3 学習動作の実行手順

### 3.1 環境変数設定

```
$ export PYTHONPATH=~/work/fast-rcnn/caffe-fast-rcnn/python:$PYTHONPATH
```


### 3.2 ダミーmatlabコマンド配置

```
$ sudo touch /usr/local/bin/matlab
$ sudo chmod a+x /usr/local/bin/matlab
```


### 3.3 学習動作のスクリプトを実行

```
$ cd fast-rcnn/
$ ./experiments/scripts/default_caffenet.sh cpu --iters 10
```


  --iters 10 で 10回の動作を指定
  --iters xxx 省略時のデフォルト設定は 40,000回  

------------------------------

## 4 認識動作の起動手順

### 4.1 環境変数を設定 

 必要であれば変数DISPLAYを設定

```
$ export DISPLAY=<remote host>:0
```


### 4.2 モデルファイルを配置

```
$ cd fast-rcnn
```

 学習で生成されたモデルファイルへのシンボリック・リンクを配置

```
$ (cd data/fast_rcnn_models/ ; \
ln -s ../../output/default/voc_2007_trainval/caffenet_fast_rcnn_iter_10.caffemodel \
caffenet_fast_rcnn_iter_40000.caffemodel )
```


 サイトからDownloadした40,000回の学習済のモデルファイルで試す場合は

```
$ (cd data/fast_rcnn_models/ ; ln -s ../../../Ubuntu14-set/caffenet_fast_rcnn_iter_40000.caffemodel )
```


### 4.3 認識動作のスクリプトを実行

```
$ cd fast-rcnn

$ ./tools/demo.py --net caffenet --cpu
```

以上
