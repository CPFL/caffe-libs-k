#include <vector>
#include <omp.h>
#include <boost/thread.hpp>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/timer.hpp"

static Timer timer;

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;
  }
  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    col_buffer_.Reshape(1, kernel_dim_, height_, width_);
    caffe_set(col_offset_, Dtype(0), col_buffer_.mutable_cpu_data());
    col_buffer_forward_.Reshape(1, kernel_dim_, height_, width_);
    caffe_set(col_offset_, Dtype(0), col_buffer_forward_.mutable_cpu_data());
  } else {
    col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
    caffe_set(col_offset_, Dtype(0), col_buffer_.mutable_cpu_data());
    col_buffer_forward_.Reshape(1, kernel_dim_, height_out_, width_out_);
    caffe_set(col_offset_, Dtype(0), col_buffer_forward_.mutable_cpu_data());
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
      const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  timer.Start(12);
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  timer.Stop(12, 100, "im2col");
  timer.Start(13);
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			  conv_out_channels_ / group_,
			  conv_out_spatial_dim_,
			  kernel_dim_ / group_,
			  (Dtype)1.,
			  weights + weight_offset_ * g,
			  col_buff + col_offset_ * g,
			  (Dtype)0.,
			  output + output_offset_ * g);
  }
  timer.Stop(13, 100, "blas");
}

#define im2col_partial(im_data, col_data, curr_height)			\
  for (int c = 0; c < kernel_dim_; ++c) {				\
    int w_offset = c % kernel_w_;					\
    int h_offset = (c / kernel_w_) % kernel_h_;				\
    int c_im = c / kernel_h_ / kernel_w_;				\
    int h_pad = (curr_height) * stride_h_ - pad_h_ + h_offset;		\
    for (int w = width_out_ - 1; w >= 0; --w) {				\
      int w_pad = w * stride_w_ - pad_w_ + w_offset;			\
      if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_) \
	col_data[(curr_height)*kernel_dim_*width_out_ + c*width_out_ + w] = \
	  im_data[(c_im * height_ + h_pad) * width_ + w_pad];		\
      else								\
	col_data[(curr_height)*kernel_dim_*width_out_ + c*width_out_ + w] = 0; \
    }									\
  }
    //for (int w = 0; w < width_out_; ++w) {				\

#define gemm_partial(col_data, output_data, curr_height, h_size)	\
  caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans,				\
		    conv_out_channels_ / group_,			\
		    conv_out_spatial_dim_*(h_size)/height_out_,		\
		    kernel_dim_ / group_,				\
		    (Dtype)1.,						\
		    weights + weight_offset_ * g, kernel_dim_/group_,	\
		    col_data + col_offset_/height_out_*(curr_height)	\
		    + col_offset_ * g,					\
		    conv_out_spatial_dim_*(h_size)/height_out_,		\
		    (Dtype)1.,						\
		    output_data + width_out_*(curr_height)		\
		    + output_offset_ * g, conv_out_spatial_dim_);

#define bias_partial(top_data, bias, h)					\
  for (int c = 0; c < conv_out_channels_; ++c) {			\
    for (int w = 0; w < width_out_; ++w) {				\
      int index = c * conv_out_spatial_dim_ + width_out_ * (h) + w;	\
      top_data[index] = bias[c];					\
    }									\
  }

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_pipelined(const Dtype* input,
   const Dtype* weights, Dtype* output, const Dtype* bias, int num, bool skip_im2col) {
  for (int g = 0; g < group_; ++g) {
    //timer.Start(1);
    boost::mutex mutex[omp_get_max_threads()];
#pragma omp parallel
    {
      int th = omp_get_thread_num();
      int ths = omp_get_num_threads();
      bool is_group1 = th < ths/2;
      int th2;
      if (is_group1) {
	th2 = th + ths / 2;
      } else {
	th2 = th - ths / 2;
      }
      Dtype* col_buff = col_buffer_.mutable_cpu_data();
      const Dtype* input_data = input;
      Dtype* output_data = output;
      int h_start;
      int h_end;
      if (height_out_%(ths/2) == 0) {
	h_start	= height_out_/(ths/2)*(th%(ths/2));
	h_end = height_out_/(ths/2)*(th%(ths/2)+1);
      } else {
	h_start = (height_out_/(ths/2)+1)*(th%(ths/2));
	h_end = (height_out_/(ths/2)+1)*(th%(ths/2)+1);
      }
      // LOG(INFO) << "thread " << th << " : " << h_start << " to " << h_end;
      //printf("%d %d\n", h_start, h_end);
      int input_size = height_ * width_ * conv_in_channels_;
      int output_size = height_out_ * width_out_ * conv_out_channels_;
      //if (th == 0) timer.Start(3);
      if (!is_group1) {
	if (h_start < height_out_) {
	  im2col_partial(input_data, col_buff, h_start);
	  bias_partial(output_data, bias, h_start);
	}
	h_start++;
      } else {
	h_end--;
      }
#pragma omp barrier
      //if (th == 0) timer.Stop(3, 1000, "convcol1");
      for (int n = 0; n < num; ++n) {
	for (int h = h_start; h < h_end; ++h) {
	  // if (th == 0 && height_==120) timer.Start(12);
	  if (is_group1) {
	    // if (th == 0 && height_==120) timer.Start(21);
	    mutex[th].lock();
	    if (h < height_out_) {
	      gemm_partial(col_buff, output_data, h, 1);
	    }
	    mutex[th].unlock();
	    mutex[th2].lock();
	    mutex[th2].unlock();
	    // if (th == 0 && height_==120) timer.Stop(21, 1000, "convblas");
	  } else {
	    // if (th == 8 && height_==120) timer.Start(20);
	    mutex[th].lock();
	    if (h < height_out_) {
	      im2col_partial(input_data, col_buff, h);
	      bias_partial(output_data, bias, h);
	    }
	    mutex[th].unlock();
	    mutex[th2].lock();
	    mutex[th2].unlock();
	    // if (th == 8 && height_==120) timer.Stop(20, 1000, "convcol2");
	  }
	  // if (th == 0 && height_==120) timer.Stop(12, 200, "conv100");
#pragma omp barrier
	}
	// if (th == 0 && height_==121) timer.Stop(12, 100, "conv100");
	input_data += input_size;
	if (n < num - 1) {
	  if (is_group1) {
	    mutex[th].lock();
	    if (h_end < height_out_) {
	      gemm_partial(col_buff, output_data, h_end, 1);
	    }
	    mutex[th].unlock();
	    mutex[th2].lock();
	    mutex[th2].unlock();
	  } else {
	    mutex[th].lock();
	    if (h_start - 1 < height_out_) {
	      im2col_partial(input_data, col_buff, h_start - 1);
	      Dtype* o = output_data + output_size;
	      bias_partial(o, bias, h_start - 1);
	    }
	    mutex[th].unlock();
	    mutex[th2].lock();
	    mutex[th2].unlock();
	  }
#pragma omp barrier
	} else {
	  if (h_end <= height_out_) {
	    if (is_group1) {
	      gemm_partial(col_buff, output_data, h_end, 1);
	    }
	  }
	}
	output_data += output_size;
      }
    }
    //printf("%d : %d\n", height_out_, counter);
    //timer.Stop(1, 1000, "conv");
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

#define gemm_partial2(curr_height, h_size)				\
  caffe_gemm<Dtype>(CblasTrans, CblasNoTrans,				\
		    kernel_dim_ / group_,				\
		    conv_out_spatial_dim_*(h_size)/height_out_,		\
		    conv_out_channels_ / group_,			\
		    (Dtype)1.,						\
		    weights + weight_offset_ * g,			\
		    kernel_dim_ / group_,				\
		    output_data +width_out_*(curr_height) + output_offset_ * g, \
		    conv_out_spatial_dim_,				\
		    (Dtype)0.,						\
		    col_buff + width_out_*(curr_height) + col_offset_ * g, \
		    conv_out_spatial_dim_);

#define col2im_partial(data_col, data_im, h)				\
  int height_col = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;	\
  int width_col = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;	\
  int channels_col = conv_in_channels_ * kernel_h_ * kernel_w_;		\
  for (int c = 0; c < channels_col; ++c) {				\
    int w_offset = c % kernel_w_;					\
    int h_offset = (c / kernel_w_) % kernel_h_;				\
    int c_im = c / kernel_h_ / kernel_w_;				\
    int h_pad = (h) * stride_h_ - pad_h_ + h_offset;			\
    for (int w = 0; w < width_col; ++w) {				\
      int w_pad = w * stride_w_ - pad_w_ + w_offset;			\
      if (h_pad >= 0 && h_pad < height_ && w_pad >= 0 && w_pad < width_) \
	data_im[(c_im * height_ + h_pad) * width_ + w_pad] +=		\
	  data_col[(c * height_col + (h)) * width_col + w];		\
    }									\
  }

  /*
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_pipelined(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  caffe_set(height_ * width_ * conv_in_channels_ * num_, Dtype(0), input);
  for (int g = 0; g < group_; ++g) {
    int counter = 0;
#pragma omp parallel
    {
      int th = omp_get_thread_num();
      int ths = omp_get_num_threads();
      Dtype* col_buff = col_buffer_.mutable_cpu_data();
      Dtype* input_data = input;
      const Dtype* output_data = output;
      int h_start = height_out_/(ths/2)*(th%4);
      int h_end = height_out_/(ths/2)*(th%4+1);
      int input_size = height_ * width_ * conv_in_channels_;
      int output_size = height_out_ * width_out_ * conv_out_channels_;
      //if (th == 0) timer.Start(3);
      gemm_partial2(h_start, 1);
      //if (th == 0) timer.Stop(3, 1000, "convcol1");
      for (int n = 0; n < num_; ++n) {
	for (int h = h_start; h < h_end-1; ++h) {
	  if (th < 8) {
	    //if (th == 0 && height_ == 60) timer.Start(14);
	    col2im_partial(col_buff, input_data, h);
	    //if (th == 0 && height_ == 60) timer.Stop(14, 1000, "backcol2");
	  } else {
	    //if (th == 4 && height_ == 60) timer.Start(15);
	    gemm_partial2(h+1, 1);
	    //if (th == 4 && height_ == 60) timer.Stop(15, 1000, "backblas");
	  }
#pragma omp barrier
	}
	input_data += input_size;
	if (n < num_-1) {
	  if (th < 8) {
	    col2im_partial(col_buff, input_data, h_end-1);
	    output_data += output_size;
	  } else {
	    output_data += output_size;
	    gemm_partial2(h_start, 1);
	  }
#pragma omp barrier
	} else {
	  if (th < 8) {
	    col2im_partial(col_buff, input_data, h_end-1);
	  }
	}
      }
    }
  }
}
  */
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  //timer.Start(1);
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  //timer.Stop(1, 100, "backblas");
  //timer.Start(0);
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
  //timer.Stop(0, 100, "backcol");
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  //timer.Start(2);
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  //timer.Stop(2, 100, "weightcol");
  //timer.Start(3);
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
  //timer.Stop(3, 100, "weightblas");
}

#define gemm_partial3(col_data, output_data, curr_height, h_size)	\
  caffe_gemm<Dtype>(CblasNoTrans, CblasTrans,				\
		    conv_out_channels_ / group_,			\
		    kernel_dim_ / group_,				\
		    conv_out_spatial_dim_*(h_size)/height_out_,		\
		    (Dtype)1.,						\
		    output,						\
		    col_data + col_offset_/height_out_*(curr_height)	\
		    + col_offset_ * g,					\
		    kernel_dim_/group_,					\
		    conv_out_spatial_dim_*(h_size)/height_out_,		\
		    (Dtype)1.,						\
		    output_data + width_out_*(curr_height)		\
		    + output_offset_ * g,				\
		    conv_out_spatial_dim_);
  /*    
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_pipelined(const Dtype* input, const Dtype* output, Dtype* weight, int num) {
  //int h_size = 2;
  for (int g = 0; g < group_; ++g) {
    //timer.Start(1);
    //int counter = 0;
#pragma omp parallel
    {
      int th = omp_get_thread_num();
      int ths = omp_get_num_threads();
      bool is_group1 = th < ths/2;
      //printf("%d:%d\n", th, ths);
      Dtype* col_buff = col_buffer_.mutable_cpu_data();
      const Dtype* input_data = input;
      Dtype* output_data = weight;
      int h_start;
      int h_end;
      if (height_out_%(ths/2) == 0) {
	h_start	= height_out_/(ths/2)*(th%8);
	h_end = height_out_/(ths/2)*(th%8+1);
      } else {
	h_start = (height_out_/(ths/2)+1)*(th%8);
	h_end = (height_out_/(ths/2)+1)*(th%8+1);
      }
      
      //printf("%d %d\n", h_start, h_end);
      int input_size = height_ * width_ * conv_in_channels_;
      int output_size = height_out_ * width_out_ * conv_out_channels_;
      //if (th == 0) timer.Start(3);
      im2col_partial(input_data, col_buff, h_start);
      bias_partial(output_data, bias, h_start);
#pragma omp barrier
      //if (th == 0) timer.Stop(3, 1000, "convcol1");
      for (int n = 0; n < num; ++n) {
	if (th == 0 && height_==121) timer.Start(12);
	for (int h = h_start; h < h_end-1; ++h) {
	  //if (th == 0 && height_==120) timer.Start(12);
	  //printf("%d:%d\n", n, h);
	  if (is_group1) {
	    if (h <= height_out_-1) {
	      //if (th == 0 && height_==30) timer.Start(21);
	      gemm_partial(col_buff, output_data, h, 1);
	      //if (th == 0 && height_==30) timer.Stop(21, 2000, "convblas");
	    }
	  } else {
	    if (h+1 <= height_out_-1) {
	      //if (th == 8 && height_==30) timer.Start(20);
	      im2col_partial(input_data, col_buff, h+1);
	      bias_partial(output_data, bias, h+1);
	      //if (th == 8 && height_==30) timer.Stop(20, 2000, "convcol2");
	    }
	  }
#pragma omp barrier
	  //if (th == 0 && height_==120) timer.Stop(12, 1000, "conv100");
	}
	if (th == 0 && height_==121) timer.Stop(12, 100, "conv100");
	input_data += input_size;
	if (n < num-1) {
	  if (is_group1) {
	    if (h_end <= height_out_) {
	      gemm_partial(col_buff, output_data, h_end-1, 1);
	    }
	  } else {
	    im2col_partial(input_data, col_buff, h_start);
	    Dtype* o = output_data+output_size;
	    bias_partial(o, bias, h_start);
	  }
#pragma omp barrier
	} else {
	  if (h_end-1 <= height_out_) {
	    if (is_group1) {
	      gemm_partial(col_buff, output_data, h_end-1, 1);
	    }
	  }
	}
	output_data += output_size;
      }
    }
    //printf("%d : %d\n", height_out_, counter);
    //timer.Stop(1, 1000, "conv");
  }
}
  */
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
