#include "caffe/util/timer.hpp"

void Timer::Start(int i)
{
  start[i] = MPI_Wtime();
}

void Timer::Stop(int i, int iter, std::string name)
{
  double t = MPI_Wtime() - start[i];
  if (count[i] == 0) {
    time[i] = t;
    min[i] = t;
    max[i] = t;
  } else {
    time[i] += t;
    if (t < min[i]) min[i] = t;
    if (t > max[i]) max[i] = t;
  }
  
  count[i]++;
  
  
  if (count[i] >= iter) {
    //printf("%s :%fs\n",name, time/100);
    std::cout << std::setw(8) << name << ":: ";
    printf("iter:%d ave:%lf min:%lf max:%lf\n", count[i], time[i]/iter, min[i], max[i]);
    //LOG(INFO) << name << ":" << time[i]/iter << "s";
    count[i] = 0;
  }
}
