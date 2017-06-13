#include <string>
#include <vector>
//#include <boost/thread.hpp>

#include <boost/make_shared.hpp>

#include "caffe/util/mpi.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/mpi_functions.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/timer.hpp"

namespace caffe {
namespace MPI_ {

static Timer timer;

void Init() {
  MPI_Init(NULL, NULL);
}

void Init_thread(int req, int* prov) {
  int thread_ = MPI_Init_thread(NULL, NULL, req, prov);
}

void Barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
}

void Finalize() {
  MPI_Finalize();
}

int rank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int comm_size() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

template <typename Dtype>
void Scheduler<Dtype>::EntryRequests(const int size, Dtype* const data, const string& layer_name) {
  if (reqs_.size() == 0) {
    vector<RequestParams> params;
    Request req = {boost::make_shared<vector<RequestParams> >(params), layer_name};
    reqs_.push_back(req);
  } else {
    if (layer_name != reqs_.back().name) {
      vector<RequestParams> params;
      Request req = {boost::make_shared<vector<RequestParams> >(params), layer_name};
      reqs_.push_back(req);
    }
  }
  RequestParams param = {data, NULL, size, boost::shared_ptr<AtomicFlag>()};
  param.unoccupied = boost::make_shared<AtomicFlag>(new AtomicFlag(true));
  reqs_.back().params->push_back(param);
  reqs_.back().params->back().unoccupied = boost::make_shared<AtomicFlag>(new AtomicFlag(true));
  LOG(INFO) << "Entry: " << reqs_.back().name << " size: " << reqs_.back().params->back().size;
}

template <typename Dtype>
void Scheduler<Dtype>::Start() {
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void Scheduler<Dtype>::SendRequest(const string& layer_name, int param_num) {
  RequestParams& param = (*GetParamsOf(layer_name))[param_num];
  //LOG(INFO) << rank() << "Req " << layer_name;
  param.unoccupied->Set(false);
}

template <typename Dtype>
bool Scheduler<Dtype>::CheckRequest(const string& layer_name) {
  const vector<RequestParams>* params = GetParamsOf(layer_name);
  //LOG(INFO) << rank() << "Check" << layer_name;// << " " << (*params)[0].unoccupied->Get();
  if (params == NULL) {
    return true;
  }
  bool rv = true;
  for (int i = 0; i < (*params).size(); ++i) {
    bool unoccupied = (*params)[i].unoccupied->Get();
    if (!unoccupied) {
      rv = false;
    }
  }
  return rv;
}

template <typename Dtype>//TODO
void Scheduler<Dtype>::Wait() {
  for (int i = 0; i < reqs_.size(); ++i) {
    while (true) {
      bool flag = true;
      for (int j = 0; j < reqs_[i].params->size(); ++j) {
	if (!reqs_[i].params->at(j).unoccupied->Get()) {
	  flag = false;
	}
      }
      if (flag) break;
    }
  }
}

template <typename Dtype>
void Scheduler<Dtype>::stopInstance() {
  stop_.Set(true);
}

template <typename Dtype>
void Scheduler<Dtype>::InternalThreadEntry() {
  LOG(INFO) << "Thread " << boost::this_thread::get_id() << " is used for scheduling.";
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  sched_setaffinity(0, sizeof(set), &set);
  while (true) {
    RequestParams* param = NULL;
    int target;
    for (int i = reqs_.size()-1; i >= 0; --i) {
      bool flag = false;
      for (int j = 0; j < reqs_[i].params->size(); ++j) {
	while (reqs_[i].params->at(j).unoccupied->Get()) {
	  if (stop_.Get()) {
	    return;
	  }
	}
	param = &reqs_[i].params->at(j);
	target = i;
	//LOG(INFO) << rank() << " Reduce start " << reqs_[target].name;
	if (rank() == 0) timer.Start(target*2+j);
	caffe_Allreduce<Dtype>(param->data, param->data, param->size, MPI_SUM);
	if (rank() == 0) timer.Stop(target*2+j, 50, string("MPI") + reqs_[target].name);
	//LOG(INFO) << rank() << " Reduce end " << reqs_[target].name;
	param->unoccupied->Set(true);
      }
    }
  }
}

INSTANTIATE_CLASS(Scheduler);

} // namespace MPI_

} // namespace caffe
