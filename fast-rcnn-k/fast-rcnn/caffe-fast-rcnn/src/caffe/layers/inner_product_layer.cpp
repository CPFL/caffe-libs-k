#include <vector>
#include <omp.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/mpi.hpp"
#include "caffe/util/mpi_functions.hpp"
#include "caffe/util/timer.hpp"

namespace caffe {

static Timer timer;

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    if (this->layer_param_.sharing_method() == string("model")) {
      vector<int> bias_shape(1, M_*MPI::comm_size());
      bias_multiplier_.Reshape(bias_shape);
      caffe_set(M_*MPI::comm_size(), Dtype(1), bias_multiplier_.mutable_cpu_data());
    } else {
      vector<int> bias_shape(1, M_);
      bias_multiplier_.Reshape(bias_shape);
      caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }
  }
  if (this->layer_param_.sharing_method() == string("model")) {
    top_shape[0] = M_*MPI::comm_size();
    top_.Reshape(top_shape);
    vector<int> bot_shape = bottom[0]->shape();
    bot_shape[0] = M_*MPI::comm_size();
    bottom_.Reshape(bot_shape);
  }
}
  template <typename Dtype>
  void InnerProductLayer<Dtype>::Forward_cpu_model_parallelized(const vector<Blob<Dtype>*>& bottom,
							       const vector<Blob<Dtype>*>& top) {
    Dtype* bottom_data = bottom[0]->mutable_cpu_data();
    Dtype* bottom_data_ = bottom_.mutable_cpu_data();
    Dtype* top_data_ = top_.mutable_cpu_data();
    int sendcount = bottom[0]->count();
    caffe_Allgather<Dtype>(bottom_data, bottom_data_, sendcount);
    MPI_Barrier(MPI_COMM_WORLD);
    int M__ = M_*MPI::comm_size();
    
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M__, N_, K_, (Dtype)1.,
			  bottom_data_, weight, (Dtype)0., top_data_);
    if (bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M__, N_, 1, (Dtype)1.,
			    bias_multiplier_.cpu_data(),
			    this->blobs_[1]->cpu_data(), (Dtype)1., top_data_);
    }
    caffe_copy(top[0]->count(), top_data_, top[0]->mutable_cpu_data());
  }
  template <typename Dtype>
  void InnerProductLayer<Dtype>::Backward_cpu_model_parallelized(const vector<Blob<Dtype>*>& top,
								const vector<bool>& propagate_down,
								const vector<Blob<Dtype>*>& bottom) {

    //if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top_.cpu_data();
    const Dtype* bottom_data = bottom_.cpu_data();
    const int M__ = M_*MPI::comm_size();
    if (this->param_propagate_down_[0]) {
      // Gradient with respect to weight
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M__, (Dtype)1.,
			    top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
      //this->blobs_[0]->Iallreduce(string("diff"));
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      //const Dtype* top_diff = top[0]->cpu_diff();
      // Gradient with respect to bias
      caffe_cpu_gemv<Dtype>(CblasTrans, M__, N_, (Dtype)1., top_diff,
			    bias_multiplier_.cpu_data(), (Dtype)0.,
			    this->blobs_[1]->mutable_cpu_diff());
      //this->blobs_[1]->Iallreduce(string("diff"));
    }
    if (propagate_down[0]) {
      //const Dtype* top_diff = top[0]->cpu_diff();
      // Gradient with respect to bottom data
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M__, K_, N_, (Dtype)1.,
			    top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
			    bottom_.mutable_cpu_diff());
      caffe_copy(bottom[0]->count(), bottom_.cpu_diff(), bottom[0]->mutable_cpu_diff());
    }
  }
  /*
template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu_model_parallelized(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //bottom[0]->Ialltoall(string("data"));
  Dtype* bottom_data_ = bottom[0]->mutable_cpu_data();
  int sendcount = bottom[0]->count()/M_/MPI::comm_size();
  int bottom_offset = bottom[0]->offset(1);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    caffe_Alltoall<Dtype>(bottom_data_, sendcount);
    bottom_data_ += bottom_offset;
  }
  Dtype* top_data_ = top_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int rank = MPI::rank();
  int comm_size = MPI::comm_size();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data()+this->blobs_[0]->count()/comm_size*rank;
  if (K_%comm_size != 0) {
    LOG(FATAL) << "Can't divide error in Inner Product Layer";
  }
  for (int i = 0; i < comm_size; ++i) {
    caffe_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_/comm_size, (Dtype)1.,
		      bottom_data+K_/comm_size*i, K_, weight, K_/comm_size, (Dtype)0.,
		      top_data_+top[0]->count()*i, N_);
  }
  //MPI_Request reqs[comm_size];
  //for (int i = 0; i < comm_size; ++i) {
    //caffe_Ireduce<Dtype>(top_.cpu_data()+i*top[0]->count(),
    //			 top_data, top[0]->count(), MPI_SUM, i, &reqs[i]);
    //}
  caffe_Allreduce<Dtype>(top_.mutable_cpu_data(), top_.mutable_cpu_data(),
  	      top[0]->count()*comm_size, MPI_SUM);
  caffe_copy(top[0]->count(), top_.cpu_data()+top[0]->count()*rank, top[0]->mutable_cpu_data());
  //for (int i = 0; i < comm_size; ++i) {
  //  caffe_Wait(&reqs[i]);
  //}
  //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
  //    bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
	this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}
  
template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu_model_parallelized(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int comm_size = MPI::comm_size();
  int rank = MPI::rank();
  caffe_copy(top[0]->count(), top[0]->cpu_diff(), top_.mutable_cpu_diff() + top[0]->count()*rank);
  MPI::Scheduler<Dtype>& scheduler = MPI::Scheduler<Dtype>::getInstance();
  scheduler.Wait();
  //#pragma omp parallel for
  for (int i = 0; i < comm_size; ++i) {
    caffe_Bcast<Dtype>(top_.mutable_cpu_diff() + top[0]->count() * i,
			top[0]->count(), i);
  }
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top_.cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_/comm_size, M_, (Dtype)1.,
		      top_diff, N_, bottom_data, K_, (Dtype)0.,
		      this->blobs_[0]->mutable_cpu_diff()+K_*N_/comm_size*rank, K_/comm_size);
    for (int i = 1; i < comm_size; ++i) {
      caffe_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_/comm_size, M_, (Dtype)1.,
			top_diff + N_*M_*i, N_, bottom_data + K_/comm_size*i, K_, (Dtype)1.,
			this->blobs_[0]->mutable_cpu_diff()+K_*N_/comm_size*rank, K_/comm_size);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    //const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_diff = top_.cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_*comm_size, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top_.cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // Gradient with respect to bottom data
    for (int i = 0; i < comm_size; ++i) {
      caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_/comm_size, N_, (Dtype)1.,
			top_diff + M_*N_*i, N_, this->blobs_[0]->cpu_data()+K_*N_/comm_size*rank,
			K_/comm_size, (Dtype)0.,
			bottom_diff + K_/comm_size*i, K_);
    }
    //bottom[0]->Ialltoall(string("diff"));
    int sendcount = bottom[0]->count()/M_/MPI::comm_size();
    int bottom_offset = bottom[0]->offset(1);
    for (int i = 0; i < bottom[0]->num(); ++i) {
      caffe_Alltoall<Dtype>(bottom_diff, sendcount);
      bottom_diff += bottom_offset;
    }
        for (int i = 0; i < comm_size; ++i) {
      Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
      caffe_Ibcast<Dtype>(weight_diff + this->blobs_[0]->count()/comm_size * i,
			  this->blobs_[0]->count()/comm_size, i, &reqs[i]);
    }
    for (int i = 0; i < comm_size; ++i) {
      caffe_Wait(&reqs[i]);
      }
  }
}
  */

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu_data_parallelized(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu_data_parallelized(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  MPI::Scheduler<Dtype>& scheduler = MPI::Scheduler<Dtype>::getInstance();
  const string& layer_name = this->layer_param_.name();

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
    if (MPI::rank() == 0 && layer_name == string("ip1")) {
      timer.Start(0);
    }
    scheduler.SendRequest(layer_name, 0);
    if (MPI::rank() == 0 && layer_name == string("ip1")) {
      timer.Stop(0, 10, string("ip1send1"));
    }
    //this->blobs_[0]->Iallreduce(string("diff"));
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
    if (MPI::rank() == 0 && layer_name == string("ip1")) {
      timer.Start(1);
    }
    scheduler.SendRequest(layer_name, 1);
    if (MPI::rank() == 0 && layer_name == string("ip1")) {
      timer.Stop(1, 10, string("ip1send2"));
    }
    //this->blobs_[1]->Iallreduce(string("diff"));
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
