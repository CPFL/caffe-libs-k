#!/bin/sh
#PJM -L "node=1"
#PJM -L "elapse=3:00:00"
#PJM -j
#PJM -S

cd boost_1_55_0
./bootstrap.sh --without-icu --with-toolset=fujitsu --prefix=$HOME/install
./b2 --disable-icu --prefix=$HOME/install --with-system --with-atomic --with-python --with-regex --with-date_time --with-iostreams --with-thread --toolset=fujitsu -j8 install
