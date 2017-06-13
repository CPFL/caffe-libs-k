#ifndef CAFFE_UTIL_MPI_HPP_
#define CAFFE_UTIL_MPI_HPP_

//#undef SEEK_SET
//#undef SEEK_END
//#undef SEEK_CUR
#include <mpi.h>
#include <vector>
#include <string>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

namespace MPI_ {
  void Init();
  void Init_thread(int req, int* prov);
  void Finalize();
  int rank();
  int comm_size();

  class AtomicFlag {
  public:
    AtomicFlag(bool flag)
      : flag_(flag) {}
    bool Get() {
      boost::shared_lock<boost::shared_mutex> read_lock(mutex_);
      bool rv = flag_;
      return rv;
    }
    void Set(const bool flag) {
      boost::upgrade_lock<boost::shared_mutex> up_lock(mutex_);
      boost::upgrade_to_unique_lock<boost::shared_mutex> write_lock(up_lock);
      flag_ = flag;
    }
    
  private:
    bool flag_;
    boost::shared_mutex mutex_;
  };
  
  template <typename Dtype>
  class Scheduler : public InternalThread {
  private:
    explicit Scheduler()
      : stop_(false) {}
    virtual ~Scheduler() {}
    
    struct RequestParams {
      Dtype* data;
      Dtype* buff;
      int size;
      boost::shared_ptr<AtomicFlag> unoccupied;
    };
    
    struct Request {
      boost::shared_ptr<vector<RequestParams> > params;
      string name;
    };
    
    virtual void InternalThreadEntry();
    vector<RequestParams>* GetParamsOf(const string& name) {
      for (int i = reqs_.size()-1; i >= 0; --i) {
	if (reqs_[i].name == name) {
	  return reqs_[i].params.get();
	}
      }
      return NULL;
      //LOG(FATAL) << "Unknown layer name :" << name;
    }
    vector<Request> reqs_;
    AtomicFlag stop_;

  public:
    static Scheduler<Dtype>& getInstance() {
      static Scheduler<Dtype> scheduler;
      return scheduler;
    }
    
    void Start();
    void stopInstance();
    void EntryRequests(const int size, Dtype* const data, const string& layer_name);
    void SendRequest(const string& layer_name, int param_num);
    bool CheckRequest(const string& layer_name);
    void Wait();
  };
}

namespace MPI = MPI_;

}

#endif
