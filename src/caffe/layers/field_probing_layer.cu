#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/field_probing_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FieldProbingForward(const int num_samples, const int num_sliding, const int batch_size,
    const int field_dim, const Dtype step, const Dtype* filters, const Dtype* field_data,
    Dtype* top_data, const int len_coordinates, const int field_channels) {
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  int slided_num_samples = num_sliding*num_sliding*num_sliding*num_samples;
  // One thread per sample per sliding position
  if(idx < slided_num_samples) {
    int sample_idx = idx%num_samples;
    int sliding_idx = idx/num_samples;

    int p_offset = sample_idx * len_coordinates;
    Dtype px = filters[p_offset+0];
    Dtype py = filters[p_offset+1];
    Dtype pz = filters[p_offset+2];

    int k = sliding_idx%num_sliding;
    int j = (sliding_idx/num_sliding)%num_sliding;
    int i = sliding_idx/(num_sliding*num_sliding);

    Dtype sx = px + (i + 0.5) * step;
    Dtype sy = py + (j + 0.5) * step;
    Dtype sz = pz + (k + 0.5) * step;

    Dtype x, y, z;
    int x0, y0, z0, x1, y1, z1;
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    int field_dim_1 = field_dim - 1;
    x = sx; y = sy; z = sz;
    SnapGrid_gpu(x, x0, x1, x_a, x_m, field_dim_1);
    SnapGrid_gpu(y, y0, y1, y_a, y_m, field_dim_1);
    SnapGrid_gpu(z, z0, z1, z_a, z_m, field_dim_1);

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      int top_offset = batch_idx * slided_num_samples + idx;
      Dtype* t_data = top_data + top_offset * field_channels;
      Interpolate_gpu(field_data, batch_idx, x, y, z, x0, y0, z0, x1, y1, z1, field_dim, t_data, field_channels);
    } /* batch_idx */
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num_samples = num_curve_ * len_curve_;
  int slided_num_samples = num_sliding_ * num_sliding_ * num_sliding_ * num_samples;
  Dtype step = field_dim_ * 1.0 / num_sliding_;

  const Dtype* filters = this->blobs_[0]->gpu_data();
  for (int field_idx = 0; field_idx < field_num_; ++field_idx) {
    const std::vector<int>& field_shape = bottom[field_idx]->shape();
    int field_channels = (field_shape.size() == 5) ? (field_shape.back()) : (1);

    const Dtype* bottom_data = bottom[field_idx]->gpu_data();
    Dtype* top_data = top[field_idx]->mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    FieldProbingForward<Dtype><<<CAFFE_GET_BLOCKS(slided_num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, num_sliding_,
        batch_size_, field_dim_, step, filters, bottom_data, top_data, len_coordinates, field_channels);
    CUDA_POST_KERNEL_CHECK;
  } /* field_num_ */
}

template <typename Dtype>
__global__ void FieldProbingBackward(const int num_samples, const int num_sliding, const int batch_size,
    const int field_dim, const Dtype step, const Dtype* filters, Dtype* filters_diff, const Dtype* field_data,
    const Dtype* top_data, const Dtype* top_diff, const int len_coordinates, const int field_channels) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread per sample
  if(sample_idx < num_samples) {
    Dtype* gradients = new Dtype[field_channels * 3];

    int field_dim_1 = field_dim - 1;
    int slided_num_samples = num_sliding*num_sliding*num_sliding*num_samples;

    int p_offset = sample_idx * len_coordinates;
    Dtype px = filters[p_offset + 0];
    Dtype py = filters[p_offset + 1];
    Dtype pz = filters[p_offset + 2];

    int sliding_idx = 0;
    for (int i = 0; i < num_sliding; ++i) {
      Dtype sx = px + (i + 0.5) * step;
      for (int j = 0; j < num_sliding; ++j) {
        Dtype sy = py + (j + 0.5) * step;
        for (int k = 0; k < num_sliding; ++k) {
          Dtype sz = pz + (k + 0.5) * step;

          Dtype x, y, z;
          int x0, y0, z0, x1, y1, z1;
          Dtype x_a, y_a, z_a, x_m, y_m, z_m;
          x = sx; y = sy; z = sz;
          SnapGrid_gpu(x, x0, x1, x_a, x_m, field_dim_1);
          SnapGrid_gpu(y, y0, y1, y_a, y_m, field_dim_1);
          SnapGrid_gpu(z, z0, z1, z_a, z_m, field_dim_1);

          Dtype w_diff_x, w_diff_y, w_diff_z;
          w_diff_x = w_diff_y = w_diff_z = 0;

          for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            Dtype diff_x, diff_y, diff_z;
            diff_x = diff_y = diff_z = 0;
            int top_offset = batch_idx * slided_num_samples + sliding_idx * num_samples + sample_idx;
            const Dtype* p_top_diff = top_diff + top_offset * field_channels;
            ComputeGradient_gpu(field_data, batch_idx, x, y, z, x0, y0, z0, x1, y1, z1,
                x_a, y_a, z_a, x_m, y_m, z_m, field_dim, gradients, field_channels);

            for (int channel_idx = 0; channel_idx < field_channels; ++channel_idx) {
              diff_x += p_top_diff[channel_idx] * gradients[3 * channel_idx + 0];
              diff_y += p_top_diff[channel_idx] * gradients[3 * channel_idx + 1];
              diff_z += p_top_diff[channel_idx] * gradients[3 * channel_idx + 2];
            }

            w_diff_x += diff_x;
            w_diff_y += diff_y;
            w_diff_z += diff_z;
          } /* batch_idx */

          filters_diff[p_offset + 0] += w_diff_x;
          filters_diff[p_offset + 1] += w_diff_y;
          filters_diff[p_offset + 2] += w_diff_z;

          sliding_idx++;
        } /* k */
      } /* j */
    } /* i */
    delete gradients;
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num_samples = num_curve_ * len_curve_;
  int num_sliding_total = num_sliding_ * num_sliding_ * num_sliding_;
  Dtype step = field_dim_ * 1.0 / num_sliding_;

  const Dtype* filters = this->blobs_[0]->gpu_data();

  Dtype* filters_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), filters_diff);
  for (int field_idx = 0; field_idx < field_num_; ++field_idx) {
    const std::vector<int>& field_shape = bottom[field_idx]->shape();
    int field_channels = (field_shape.size() == 5) ? (field_shape.back()) : (1);

    const Dtype* bottom_data = bottom[field_idx]->gpu_data();
    const Dtype* top_data = top[field_idx]->gpu_data();
    const Dtype* top_diff = top[field_idx]->gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    FieldProbingBackward<Dtype><<<CAFFE_GET_BLOCKS(num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, num_sliding_,
        batch_size_, field_dim_, step, filters, filters_diff, bottom_data, top_data, top_diff, len_coordinates, field_channels);
    CUDA_POST_KERNEL_CHECK;
  } /* field_num_ */

  //caffe_gpu_scal(this->blobs_[0]->count(), Dtype(1.0 / (num_sliding_total * batch_size_ * field_num_)), filters_diff);
  caffe_gpu_scal(this->blobs_[0]->count(), Dtype(1.0 / (num_sliding_total * field_num_)), filters_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(FieldProbingLayer);

}  // namespace caffe
