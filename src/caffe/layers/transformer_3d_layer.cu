#include "caffe/layers/transformer_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Transformer3DForward(const int num_samples, const int batch_size,
    const Dtype* transformations, const Dtype* bottom_data, Dtype* top_data, const int len_transformation_param) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each sample
  if(sample_idx < num_samples) {
    int top_count = num_samples*batch_size;
    for (int offset = sample_idx, batch_idx = 0; offset < top_count; offset += num_samples) {
      int offsetx4 = offset*4;
      const Dtype& x = bottom_data[offsetx4+0];
      const Dtype& y = bottom_data[offsetx4+1];
      const Dtype& z = bottom_data[offsetx4+2];

      const Dtype* t = transformations+batch_idx*len_transformation_param;
      top_data[offsetx4+0] = (t[0]+1)*x + t[1]*y + t[2]*z + t[3];
      top_data[offsetx4+1] = t[4]*x + (t[5]+1)*y + t[6]*z + t[7];
      top_data[offsetx4+2] = t[8]*x + t[9]*y + (t[10]+1)*z + t[11];

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

  if(rand()%100 == 0) {
    const Dtype* t = bottom[1]->cpu_data()+len_transformation_param*(rand()%batch_size);
    LOG(INFO) << "\t" << t[0]+1 << " " << t[1] << " " << t[2] << " " << t[3] << std::endl;
    LOG(INFO) << "\t" << t[4] << " " << t[5]+1 << " " << t[6] << " " << t[7] << std::endl;
    LOG(INFO) << "\t" << t[8] << " " << t[9] << " " << t[10]+1 << " " << t[11] << std::endl;
    LOG(INFO) << "\t" << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
  }
}

template <typename Dtype>
__global__ void Transformer3DBackward(const int num_samples, const int batch_size,
    const Dtype* transformations, const Dtype* bottom_data, const Dtype* top_diff,
    Dtype* temp_diff, Dtype* bottom_diff, const int len_transformation_param) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each sample
  if(sample_idx < num_samples) {
    int top_count = num_samples*batch_size;
    for (int offset = sample_idx, batch_idx = 0; offset < top_count; offset += num_samples) {
      int offsetx4 = offset*4;
      const Dtype& x = bottom_data[offsetx4+0];
      const Dtype& y = bottom_data[offsetx4+1];
      const Dtype& z = bottom_data[offsetx4+2];

      const Dtype& x_d = top_diff[offsetx4+0];
      const Dtype& y_d = top_diff[offsetx4+1];
      const Dtype& z_d = top_diff[offsetx4+2];

      const Dtype* t = transformations+batch_idx*len_transformation_param;
      bottom_diff[offsetx4+0] = (t[0]+1)*x_d + t[4]*y_d + t[8]*z_d;
      bottom_diff[offsetx4+1] = t[1]*x_d + (t[5]+1)*y_d + t[9]*z_d;
      bottom_diff[offsetx4+2] = t[2]*x_d + t[6]*y_d + (t[10]+1)*z_d;

      Dtype* t_diff = temp_diff+(batch_idx*num_samples+sample_idx)*len_transformation_param;
      t_diff[0] = x*x_d;
      t_diff[1] = y*x_d;
      t_diff[2] = z*x_d;
      t_diff[3] = x_d;
      t_diff[4] = x*y_d;
      t_diff[5] = y*y_d;
      t_diff[6] = z*y_d;
      t_diff[7] = y_d;
      t_diff[8] = x*z_d;
      t_diff[9] = y*z_d;
      t_diff[10] = z*z_d;
      t_diff[11] = z_d;

      batch_idx ++;
    }
  }
}

template <typename Dtype>
__global__ void Transformer3DBackward(const int num_params, const int num_samples,
    const Dtype* temp_diff, Dtype* transformations_diff) {
  int param_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each param
  if(param_idx < num_params) {
    const Dtype* t_diff = temp_diff + param_idx*num_samples;
    Dtype sum = 0.0;
    for (int i = 0; i < num_samples; ++ i) {
        sum += t_diff[i];
    }
    transformations_diff[param_idx] = sum;
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
  Dtype* temp_diff = temp_diff_.mutable_gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  Transformer3DBackward<Dtype><<<CAFFE_GET_BLOCKS(num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, batch_size,
    transformations, bottom_data, top_diff, temp_diff, bottom_diff, len_transformation_param);
  CUDA_POST_KERNEL_CHECK;

  int num_params = batch_size*len_transformation_param;
  Dtype* transformations_diff = bottom[1]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  Transformer3DBackward<Dtype><<<CAFFE_GET_BLOCKS(num_params), CAFFE_CUDA_NUM_THREADS>>>(num_params, num_samples,
    temp_diff, transformations_diff);
  CUDA_POST_KERNEL_CHECK;

  caffe_gpu_scal(num_params, Dtype(1.0/num_samples), transformations_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(Transformer3DLayer);

}  // namespace caffe
