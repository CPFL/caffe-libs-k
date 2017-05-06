#include <vector>
#include <omp.h>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/mpi.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    if (this->layer_param_.convolution_param().pipeline()) {
      const Dtype* bias = this->blobs_[1]->cpu_data();
      this->forward_cpu_pipelined(bottom_data, weight, top_data, bias, this->num_);
    } else {
      for (int n = 0; n < this->num_; ++n) {
	this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
			       top_data + top[i]->offset(n));
	if (this->bias_term_) {
	  const Dtype* bias = this->blobs_[1]->cpu_data();
	  int top_size = this->height_out_*this->width_out_;
	  int out_channels = this->layer_param_.convolution_param().num_output();
	  this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
	}
      }
    }
  }
}

  /*
template <typename Dtype>
void ConvolutionLayer<Dtype>::forward_cpu_raw(const vector<Blob<Dtype>*>& bottom,
					      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      int out_channels__ = this->layer_param_.convolution_param().num_output();
      int width_out__ = this->width_out_;
      int height_out__ = this->height_out_;
      int width__ = this->width_;
      int height__ = this->height_;
      int top_size__ = width_out__ * height_out__;
      int pad_h__ = this->pad_h_;
      int pad_w__ = this->pad_w_;
      int stride_w__ = this->stride_w_;
      int stride_h__ = this->stride_h_;
      int channels__ = this->channels_;
      if (this->channels_ <= 4) {
#pragma omp parallel for
	for (int c = 0; c < 32; ++c) {
	  Dtype *top_data_ = top_data+c*top_size__;
	  for (int h = 0; h < height_out__; ++h) {
	    //for (int h_ = omp_get_thread_num()*5;
	      //h_ < height_out__+omp_get_thread_num()*5; ++h_) {
	     // int h = h_;
	      //if (h >= height_out__) h -= height_out__;
	    for (int w = 0; w < width_out__; ++w) {
	      //for (int w_ = omp_get_thread_num()*5;
		//w_ < width_out__+omp_get_thread_num()*5; ++w_) {
		//int w = w_;
		//if (w >= width_out__) w -= width_out__;
	      int w_diff = w * stride_w__ - pad_w__;
	      int h_diff = h * stride_h__ - pad_h__;
	      Dtype sum = 0;
	      int kh = 5;
	      while (--kh) {
		//for (int kh = 0; kh <= 4; ++kh) {
		int h_pos = h_diff+kh;
		if (h_pos >= 0 && h_pos < height__) {
		  int kw = 5;
		  while (--kw) {
		    //for (int kw = 0; kw <= 4; ++kw) {
		    int w_pos = w_diff+kw;
		    if (w_pos >= 0 && w_pos < width__) {
		      //for (int ch = 0; ch < 4; ++ch) {
		      int ch = channels__;
		      while (--ch) {
			//int ch=0;
			int index = ch*height__*width__ + (w_pos + h_pos*width__);
			int windex = c*25*32 + (ch*5 + kh)*5 + kw;
			//int index = 4*height__*width__ + (w_pos + h_pos*width__)*4+ch;
			//int windex = c*25*4 + (4*5 + (kh+2)*4)*5 + kw+2+ch;
			//const Dtype *pbot_data = bottom_data+index;
			//const Dtype *pweight = weight+windex;
			sum += bottom_data[index]*weight[windex];
			//for (int j = 0; j < 32; ++j) {
			//sum += *pbot_data++ + *pweight++;
			//}
		      }
		    }
		  }
		}
	      }
	      *top_data_++ = sum;
	    }
	  }
	}
*/

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  MPI::Scheduler<Dtype>& scheduler = MPI::Scheduler<Dtype>::getInstance();
  const string& layer_name = this->layer_param_.name();

  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
      }
      scheduler.SendRequest(layer_name, 0);
      //this->blobs_[0]->Iallreduce(string("diff"));
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
      scheduler.SendRequest(layer_name, 1);
      //this->blobs_[1]->Iallreduce(string("diff"));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
