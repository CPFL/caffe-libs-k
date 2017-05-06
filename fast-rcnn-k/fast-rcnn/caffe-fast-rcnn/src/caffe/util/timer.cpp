#include "caffe/util/timer.hpp"

void Timer::Start(int i)
{
  start[i] = omp_get_wtime();
}

void Timer::Stop(int i, int iter, std::string name)
{
  if (count[i] == 0) {
    time[i] = 0;
  }
  //clock_gettime(CLOCK_MONOTONIC_RAW, &stop[i]);
  stop[i] = omp_get_wtime();
  time[i] += stop[i] - start[i];
  //time[i] += ((double)stop[i].tv_sec + (double)stop[i].tv_nsec * 1e-9)
  //  - ((double)start[i].tv_sec + (double)start[i].tv_nsec * 1e-9);
  count[i]++;
  
  if (count[i] >= iter) {
    //printf("%s :%fs\n",name, time/100);
    std::cout << std::setw(8) << name << ":"; 
    printf("%lf\n", time[i]/iter);
    count[i] = 0;
  }
}

void Timer::Init(int i, const char *n)
{
  //iter = i;
  //name = n;
  //count = 0;
}
  
