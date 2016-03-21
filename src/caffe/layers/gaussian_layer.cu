#include <algorithm>
#include <vector>

#include "caffe/layers/gaussian_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GaussianForward(const int n, const Dtype* bottom_data, Dtype* top_data, Dtype* x_data, Dtype* gaussian_x_data,
    Dtype multiplier);

template <>
__global__ void GaussianForward<float>(const int n, const float* bottom_data, float* top_data, float* x_data, float* gaussian_x_data,
    float multiplier) {
  CUDA_KERNEL_LOOP(index, n) {
    const float& x = bottom_data[index];
    float gaussian_x = expf(x*x*multiplier);
    top_data[index] = gaussian_x;
    x_data[index] = x;
    gaussian_x_data[index] = gaussian_x;
  }
}

template <>
__global__ void GaussianForward<double>(const int n, const double* bottom_data, double* top_data, double* x_data, double* gaussian_x_data,
    double multiplier) {
  CUDA_KERNEL_LOOP(index, n) {
    const double& x = bottom_data[index];
    double gaussian_x = exp(x*x*multiplier);
    top_data[index] = gaussian_x;
    x_data[index] = x;
    gaussian_x_data[index] = gaussian_x;
  }
}

template <typename Dtype>
void GaussianLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* x_data = x_and_gaussian_x_.mutable_gpu_data();
  Dtype* gaussian_x_data = x_and_gaussian_x_.mutable_gpu_diff();
  const int count = bottom[0]->count();
  Dtype sigma = this->layer_param_.gaussian_param().sigma();
  Dtype multiplier = -1.0/(2.0*sigma*sigma);
  // NOLINT_NEXT_LINE(whitespace/operators)
  GaussianForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, x_data, gaussian_x_data, multiplier);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void GaussianBackward(const int n, const Dtype* top_diff,
  const Dtype* x_data, const Dtype* gaussian_x_data, Dtype* bottom_diff, Dtype multiplier) {
  CUDA_KERNEL_LOOP(index, n) {
    bottom_diff[index] = top_diff[index] * x_data[index] * gaussian_x_data[index] * multiplier;
  }
}

template <typename Dtype>
void GaussianLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* x_data = x_and_gaussian_x_.gpu_data();
    const Dtype* gaussian_x_data = x_and_gaussian_x_.gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype sigma = this->layer_param_.gaussian_param().sigma();
    Dtype multiplier = -1.0/(sigma*sigma);
    // NOLINT_NEXT_LINE(whitespace/operators)
    GaussianBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, x_data, gaussian_x_data, bottom_diff, multiplier);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(GaussianLayer);


}  // namespace caffe
