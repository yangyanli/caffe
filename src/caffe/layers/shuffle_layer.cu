#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/shuffle_layer.hpp"

namespace caffe {

const int CAFFE_MAX_SHUFFLE_AXES = 8;
__constant__ int input_shape[CAFFE_MAX_SHUFFLE_AXES];
__constant__ int bottom_axis_to_top[CAFFE_MAX_SHUFFLE_AXES];
__constant__ int output_shape[CAFFE_MAX_SHUFFLE_AXES];

template <typename Dtype>
__global__ void ShuffleForward(const int n, const int num_axes, const Dtype* bottom_data, Dtype* top_data) {
  extern __shared__ float output_indices[];
  CUDA_KERNEL_LOOP(index, n) {
    int input_offset = index;
    for (int i = num_axes-1; i >= 0; -- i) {
      output_indices[bottom_axis_to_top[i]*CAFFE_CUDA_NUM_THREADS+threadIdx.x] = input_offset%input_shape[i];
      input_offset /= input_shape[i];
    }

    int output_offset = 0;
    for (int i = 0; i < num_axes; ++i) {
      output_offset *= output_shape[i];
      output_offset += output_indices[i*CAFFE_CUDA_NUM_THREADS+threadIdx.x];
    }

    top_data[output_offset] = bottom_data[index];
  }
}

template <typename Dtype>
void ShuffleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int num_axes = bottom[0]->shape().size();
  const vector<int>& bottom_shape = bottom[0]->shape();
  cudaMemcpyToSymbol(input_shape, &bottom_shape[0], num_axes*sizeof(int));
  cudaMemcpyToSymbol(bottom_axis_to_top, &bottom_axis_to_top_[0], num_axes*sizeof(int));
  cudaMemcpyToSymbol(output_shape, &output_shape_[0], num_axes*sizeof(int));
  int shared_size = CAFFE_CUDA_NUM_THREADS*CAFFE_MAX_SHUFFLE_AXES*sizeof(int);
  // NOLINT_NEXT_LINE(whitespace/operators)
  ShuffleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, shared_size>>>(
      count, num_axes, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ShuffleBackward(const int n, const int num_axes, const Dtype* top_diff, Dtype* bottom_diff) {
  extern __shared__ float output_indices[];
  CUDA_KERNEL_LOOP(index, n) {
    int input_offset = index;
    for (int i = num_axes-1; i >= 0; -- i) {
      output_indices[bottom_axis_to_top[i]*CAFFE_CUDA_NUM_THREADS+threadIdx.x] = input_offset%input_shape[i];
      input_offset /= input_shape[i];
    }

    int output_offset = 0;
    for (int i = 0; i < num_axes; ++i) {
      output_offset *= output_shape[i];
      output_offset += output_indices[i*CAFFE_CUDA_NUM_THREADS+threadIdx.x];
    }

    bottom_diff[index] = top_diff[output_offset];
  }
}

template <typename Dtype>
void ShuffleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const int num_axes = bottom[0]->shape().size();
    const vector<int>& bottom_shape = bottom[0]->shape();
    cudaMemcpyToSymbol(input_shape, &bottom_shape[0], num_axes*sizeof(int));
    cudaMemcpyToSymbol(bottom_axis_to_top, &bottom_axis_to_top_[0], num_axes*sizeof(int));
    cudaMemcpyToSymbol(output_shape, &output_shape_[0], num_axes*sizeof(int));
    int shared_size = CAFFE_CUDA_NUM_THREADS*CAFFE_MAX_SHUFFLE_AXES*sizeof(int);
    // NOLINT_NEXT_LINE(whitespace/operators)
    ShuffleBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, shared_size>>>(
        count, num_axes, top_diff, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ShuffleLayer);


}  // namespace caffe
