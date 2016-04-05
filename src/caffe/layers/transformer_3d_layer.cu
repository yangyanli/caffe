#include "caffe/layers/transformer_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Transformer3DForward(const int num_samples, const int batch_size,
    const Dtype* transformations, const Dtype* bottom_data, Dtype* top_data, const int len_transformation_param) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each sample
  if(sample_idx < num_samples) {
    int top_count = num_samples*batch_size*4;
    for (int offset = sample_idx*4, batch_idx = 0; offset < top_count; offset += num_samples) {
      const Dtype* t = transformations+len_transformation_param*batch_idx;
      const Dtype& x = bottom_data[offset+0];
      const Dtype& y = bottom_data[offset+1];
      const Dtype& z = bottom_data[offset+2];

      top_data[offset+0] = t[0]*x + t[1]*y + t[2]*z + t[3];
      top_data[offset+1] = t[4]*x + t[5]*y + t[6]*z + t[7];
      top_data[offset+2] = t[8]*x + t[9]*y + t[10]*z + t[11];

      batch_idx ++;
    }
  }
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batch_size = bottom[0]->shape()[0];
  int probing_curves_size = bottom[0]->count(1);
  int num_samples = probing_curves_size/4;
  const Dtype* transformations = bottom[1]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  Transformer3DForward<Dtype><<<CAFFE_GET_BLOCKS(num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, batch_size,
    transformations, bottom_data, top_data, len_transformation_param);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void Transformer3DBackward(const int num_samples, const int batch_size,
    const Dtype* transformations, const Dtype* bottom_data, const Dtype* top_diff,
    Dtype* transformations_diff, Dtype* bottom_diff, const int len_transformation_param) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each sample
  if(sample_idx < num_samples) {
    int top_count = num_samples*batch_size*4;
    for (int offset = sample_idx*4, batch_idx = 0; offset < top_count; offset += num_samples) {
      const Dtype* t = transformations+len_transformation_param*batch_idx;
      const Dtype& x = bottom_data[offset+0];
      const Dtype& y = bottom_data[offset+1];
      const Dtype& z = bottom_data[offset+2];

      batch_idx ++;
    }
  }
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int batch_size = bottom[0]->shape()[0];
  int probing_curves_size = bottom[0]->count(1);
  int num_samples = probing_curves_size/4;
  const Dtype* transformations = bottom[1]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* transformations_diff = bottom[1]->mutable_gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  Transformer3DBackward<Dtype><<<CAFFE_GET_BLOCKS(num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, batch_size,
    transformations, bottom_data, top_diff, transformations_diff, bottom_diff, len_transformation_param);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(Transformer3DLayer);

}  // namespace caffe
