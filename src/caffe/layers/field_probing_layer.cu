#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/field_probing_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FieldProbingForward(const int num_samples, const int batch_size,
    const int field_dim_x, const int field_dim_y, const int field_dim_z, const int field_dim_x_1, const int field_dim_y_1, const int field_dim_z_1,
    const Dtype* probing_curves, const Dtype* bottom_data, Dtype* top_data, const int field_channels) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each sample
  if(sample_idx < num_samples) {
    int p_offset = sample_idx * 4;
    Dtype x = probing_curves[p_offset+0];
    Dtype y = probing_curves[p_offset+1];
    Dtype z = probing_curves[p_offset+2];

    int x0, y0, z0, x1, y1, z1;
    SnapGrid_gpu(x, x0, x1, field_dim_x_1);
    SnapGrid_gpu(y, y0, y1, field_dim_y_1);
    SnapGrid_gpu(z, z0, z1, field_dim_z_1);
    Dtype x_x0 = x-x0;
    Dtype y_y0 = y-y0;
    Dtype z_z0 = z-z0;
    Dtype x1_x = x1-x;
    Dtype y1_y = y1-y;
    Dtype z1_z = z1-z;
    int top_count = num_samples*batch_size;
    for (int top_offset = sample_idx, batch_idx = 0; top_offset < top_count; top_offset += num_samples) {
      Interpolate_gpu(bottom_data, batch_idx, x0, y0, z0, x1, y1, z1,
        x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, field_dim_x, field_dim_y, field_dim_z, top_data+top_offset*field_channels);
      batch_idx ++;
    }
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
__global__ void FieldProbingBackward(const int num_samples, const int batch_size,
    const int field_dim_x, const int field_dim_y, const int field_dim_z, const int field_dim_x_1, const int field_dim_y_1, const int field_dim_z_1,
    const Dtype* probing_curves, const Dtype* bottom_data, const Dtype* top_diff, Dtype* probing_curves_diff, const int field_channels) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;

  // One thread for each sample
  if(sample_idx < num_samples) {
    int p_offset = sample_idx * 4;
    Dtype x = probing_curves[p_offset+0];
    Dtype y = probing_curves[p_offset+1];
    Dtype z = probing_curves[p_offset+2];
  
    int x0, y0, z0, x1, y1, z1;
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    SnapGrid_gpu(x, x0, x1, x_a, x_m, field_dim_x_1);
    SnapGrid_gpu(y, y0, y1, y_a, y_m, field_dim_y_1);
    SnapGrid_gpu(z, z0, z1, z_a, z_m, field_dim_z_1);
    Dtype x_x0 = x-x0;
    Dtype y_y0 = y-y0;
    Dtype z_z0 = z-z0;
    Dtype x1_x = x1-x;
    Dtype y1_y = y1-y;
    Dtype z1_z = z1-z;
  
    Dtype w_diff_x = 0;
    Dtype w_diff_y = 0;
    Dtype w_diff_z = 0;
    Dtype* gradients = new Dtype[field_channels*3];
    int top_count = num_samples*batch_size;
    for (int top_offset = sample_idx, batch_idx = 0; top_offset < top_count; top_offset += num_samples) {
    ComputeGradient_gpu(bottom_data, batch_idx, x0, y0, z0, x1, y1, z1, x_a, y_a, z_a, x_m, y_m, z_m,
      x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, field_dim_x, field_dim_y, field_dim_z, gradients, field_channels);
      const Dtype& t_diff = top_diff[top_offset];
      for (int i = 0; i < field_channels; ++ i) {
        w_diff_x += t_diff*gradients[3*i+0];
        w_diff_y += t_diff*gradients[3*i+1];
        w_diff_z += t_diff*gradients[3*i+2];
      }

      batch_idx ++;
    }
    delete gradients;
    probing_curves_diff[p_offset+0] += w_diff_x;
    probing_curves_diff[p_offset+1] += w_diff_y;
    probing_curves_diff[p_offset+2] += w_diff_z;
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    int field_dim_1 = field_dim_ - 1;
  int num_samples = num_curve_ * len_curve_;
  int num_sliding_total = num_sliding_ * num_sliding_ * num_sliding_;
  int slided_num_samples = num_sliding_total * num_samples;

  const Dtype* probing_curves = this->blobs_[0]->gpu_data();
  const Dtype* trans = transform_ ? (bottom[field_num_]->gpu_data()) : (NULL);

  Dtype* filters_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  Dtype* trans_diff = transform_ ? (bottom[field_num_]->mutable_gpu_diff()) : (NULL);
  if(transform_) {
    caffe_gpu_set(bottom[field_num_]->count(), Dtype(0), trans_diff);
  }
  
  caffe_gpu_scal(this->blobs_[0]->count(), Dtype(1.0/(num_sliding_total*batch_size_*field_num_)), filters_diff);
  if (transform_) {
    caffe_gpu_scal(bottom[field_num_]->count(), Dtype(1.0/(slided_num_samples*batch_size_*field_num_)), trans_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FieldProbingLayer);

}  // namespace caffe
