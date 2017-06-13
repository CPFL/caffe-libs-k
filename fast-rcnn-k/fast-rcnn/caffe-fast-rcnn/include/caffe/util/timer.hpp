#ifndef CAFFE_UTIL_TIMER_H_
#define CAFFE_UTIL_TIMER_H_

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include "caffe/util/mpi.hpp"
#include "caffe/common.hpp"

class Timer
{
public:
  void Start(int i);
  void Stop(int i, int iter, std::string name);
private:
  double start[45], stop[45];
  double time[45], min[45], max[45];
  int count[45];
  int iter;
};

#endif
