#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/field_probing_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FieldProbingForward(const int num_samples, const int num_sliding, const int batch_size,
    const int field_dim, const Dtype step, const Dtype* filters, const Dtype* field_data,
    const Dtype* trans, Dtype* top_data,
    const int len_coordinates, const int len_trans_params, const int field_channels) {
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
    Dtype c_offset = field_dim_1/2.0;
    if (trans == NULL) {
      x = sx; y = sy; z = sz;
      SnapGrid_gpu(x, x0, x1, x_a, x_m, field_dim_1);
      SnapGrid_gpu(y, y0, y1, y_a, y_m, field_dim_1);
      SnapGrid_gpu(z, z0, z1, z_a, z_m, field_dim_1);
    } else {
      sx -= c_offset;
      sy -= c_offset;
      sz -= c_offset;
    }

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      if (trans != NULL) {
        const Dtype* t = trans + batch_idx * len_trans_params;
        x = t[0] * sx + t[1] * sy + t[2] * sz + t[3] + c_offset;
        y = t[4] * sx + t[5] * sy + t[6] * sz + t[7] + c_offset;
        z = t[8] * sx + t[9] * sy + t[10] * sz + t[11] + c_offset;
        SnapGrid_gpu(x, x0, x1, x_a, x_m, field_dim_1);
        SnapGrid_gpu(y, y0, y1, y_a, y_m, field_dim_1);
        SnapGrid_gpu(z, z0, z1, z_a, z_m, field_dim_1);
      }

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
  const Dtype* trans = transform_ ? (bottom[field_num_]->gpu_data()) : (NULL);
  for (int field_idx = 0; field_idx < field_num_; ++field_idx) {
    const std::vector<int>& field_shape = bottom[field_idx]->shape();
    int field_channels = (field_shape.size() == 5) ? (field_shape.back()) : (1);

    const Dtype* bottom_data = bottom[field_idx]->gpu_data();
    Dtype* top_data = top[field_idx]->mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    FieldProbingForward<Dtype><<<CAFFE_GET_BLOCKS(slided_num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, num_sliding_,
        batch_size_, field_dim_, step, filters, bottom_data, trans, top_data,
        len_coordinates, len_trans_params, field_channels);
    CUDA_POST_KERNEL_CHECK;
  } /* field_num_ */
}

template <typename Dtype>
__global__ void FieldProbingBackward(const int num_samples, const int num_sliding, const int batch_size,
    const int field_dim, const Dtype step, const Dtype* filters, Dtype* filters_diff, const Dtype* field_data,
    const Dtype* trans, const Dtype* top_data, const Dtype* top_diff,
    const int len_coordinates, const int len_trans_params, const int field_channels) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread per sample
  if(sample_idx < num_samples) {
    Dtype* gradients = new Dtype[field_channels * 3];

    int field_dim_1 = field_dim - 1;
    Dtype c_offset = field_dim_1/2.0;
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
          if (trans == NULL) {
            x = sx; y = sy; z = sz;
            SnapGrid_gpu(x, x0, x1, x_a, x_m, field_dim_1);
            SnapGrid_gpu(y, y0, y1, y_a, y_m, field_dim_1);
            SnapGrid_gpu(z, z0, z1, z_a, z_m, field_dim_1);
          } else {
            sx -= c_offset;
            sy -= c_offset;
            sz -= c_offset;
          }

          Dtype w_diff_x, w_diff_y, w_diff_z;
          w_diff_x = w_diff_y = w_diff_z = 0;

          for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            int trans_offset = batch_idx * len_trans_params;
            if (trans != NULL) {
              const Dtype* t = trans + trans_offset;
              x = t[0] * sx + t[1] * sy + t[2] * sz + t[3] + c_offset;
              y = t[4] * sx + t[5] * sy + t[6] * sz + t[7] + c_offset;
              z = t[8] * sx + t[9] * sy + t[10] * sz + t[11] + c_offset;
              SnapGrid_gpu(x, x0, x1, x_a, x_m, field_dim_1);
              SnapGrid_gpu(y, y0, y1, y_a, y_m, field_dim_1);
              SnapGrid_gpu(z, z0, z1, z_a, z_m, field_dim_1);
            }

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

            if (trans != NULL) {
              const Dtype* t = trans + trans_offset;
              w_diff_x = t[0]*diff_x + t[4]*diff_y + t[8]*diff_z;
              w_diff_y = t[1]*diff_x + t[5]*diff_y + t[9]*diff_z;
              w_diff_z = t[2]*diff_x + t[6]*diff_y + t[10]*diff_z;
            } else {
              w_diff_x += diff_x;
              w_diff_x += diff_y;
              w_diff_x += diff_z;
            }
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

template <typename Dtype>
__global__ void FieldProbingBackwardTrans(const int num_samples, const int num_sliding, const int batch_size,
    const int field_dim, const Dtype step, const Dtype* filters, const Dtype* field_data,
    const Dtype* trans, Dtype* trans_diff, const Dtype* top_data, const Dtype* top_diff,
    const int len_coordinates, const int len_trans_params, const int field_channels) {
  int batch_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread per batch
  if(batch_idx < batch_size) {
    Dtype* gradients = new Dtype[field_channels * 3];
    int field_dim_1 = field_dim - 1;
    Dtype c_offset = field_dim_1/2.0;
    int slided_num_samples = num_sliding*num_sliding*num_sliding*num_samples;
    int trans_offset = batch_idx * len_trans_params;

    Dtype a, b, c, tx;
    Dtype d, e, f, ty;
    Dtype g, h, i, tz;
    const Dtype* t = trans + trans_offset;
    a = t[0]; b = t[1]; c = t[2]; tx = t[3];
    d = t[4]; e = t[5]; f = t[6]; ty = t[7];
    g = t[8]; h = t[9]; i = t[10]; tz = t[11];

    Dtype da, db, dc, dtx;
    Dtype dd, de, df, dty;
    Dtype dg, dh, di, dtz;
    da = db = dc = dtx = 0.0;
    dd = de = df = dty = 0.0;
    dg = dh = di = dtz = 0.0;

    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
      int p_offset = sample_idx * len_coordinates;
      Dtype px = filters[p_offset + 0];
      Dtype py = filters[p_offset + 1];
      Dtype pz = filters[p_offset + 2];

      int sliding_idx = 0;
      for (int i = 0; i < num_sliding; ++i) {
        Dtype sx = px + (i + 0.5) * step - c_offset;
        for (int j = 0; j < num_sliding; ++j) {
          Dtype sy = py + (j + 0.5) * step - c_offset;
          for (int k = 0; k < num_sliding; ++k) {
            Dtype sz = pz + (k + 0.5) * step - c_offset;

            Dtype x = a * sx + b * sy + c * sz + tx + c_offset;
            Dtype y = d * sx + e * sy + f * sz + ty + c_offset;
            Dtype z = g * sx + h * sy + i * sz + tz + c_offset;

            int x0, y0, z0, x1, y1, z1;
            Dtype x_a, y_a, z_a, x_m, y_m, z_m;
            SnapGrid_gpu(x, x0, x1, x_a, x_m, field_dim_1);
            SnapGrid_gpu(y, y0, y1, y_a, y_m, field_dim_1);
            SnapGrid_gpu(z, z0, z1, z_a, z_m, field_dim_1);

            int top_offset = batch_idx * slided_num_samples + sliding_idx * num_samples + sample_idx;
            const Dtype* p_top_diff = top_diff + top_offset * field_channels;
            ComputeGradient_gpu(field_data, batch_idx, x, y, z, x0, y0, z0, x1, y1, z1,
                x_a, y_a, z_a, x_m, y_m, z_m, field_dim, gradients, field_channels);

            Dtype diff_x, diff_y, diff_z;
            diff_x = diff_y = diff_z = 0;
            for (int channel_idx = 0; channel_idx < field_channels; ++channel_idx) {
              diff_x += p_top_diff[channel_idx] * gradients[3 * channel_idx + 0];
              diff_y += p_top_diff[channel_idx] * gradients[3 * channel_idx + 1];
              diff_z += p_top_diff[channel_idx] * gradients[3 * channel_idx + 2];
            }

            da += x*diff_x; db += y*diff_x; dc += z*diff_x; dtx += diff_x;
            dd += x*diff_y; de += y*diff_y; df += z*diff_y; dty += diff_y;
            dg += x*diff_z; dh += y*diff_z; di += z*diff_z; dtz += diff_z;

            sliding_idx++;
          } /* k */
        } /* j */
      } /* i */
    } /* num_samples */
    delete gradients;

    Dtype* t_diff = trans_diff + trans_offset;
    t_diff[0] += da; t_diff[1] += db; t_diff[2] += dc; t_diff[3] += dtx;
    t_diff[4] += dd; t_diff[5] += de; t_diff[6] += df; t_diff[7] += dty;
    t_diff[8] += dg; t_diff[9] += dh; t_diff[10] += di; t_diff[11] += dtz;
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num_samples = num_curve_ * len_curve_;
  int num_sliding_total = num_sliding_ * num_sliding_ * num_sliding_;
  int slided_num_samples = num_sliding_total * num_samples;
  Dtype step = field_dim_ * 1.0 / num_sliding_;

  const Dtype* filters = this->blobs_[0]->gpu_data();
  const Dtype* trans = transform_ ? (bottom[field_num_]->gpu_data()) : (NULL);

  Dtype* filters_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), filters_diff);
  Dtype* trans_diff = NULL;
  if (transform_) {
    trans_diff = bottom[field_num_]->mutable_gpu_diff();
    caffe_gpu_set(bottom[field_num_]->count(), Dtype(0), trans_diff);
  }
  for (int field_idx = 0; field_idx < field_num_; ++field_idx) {
    const std::vector<int>& field_shape = bottom[field_idx]->shape();
    int field_channels = (field_shape.size() == 5) ? (field_shape.back()) : (1);

    const Dtype* bottom_data = bottom[field_idx]->gpu_data();
    const Dtype* top_data = top[field_idx]->mutable_gpu_data();
    const Dtype* top_diff = top[field_idx]->mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    FieldProbingBackward<Dtype><<<CAFFE_GET_BLOCKS(num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, num_sliding_,
        batch_size_, field_dim_, step, filters, filters_diff, bottom_data, trans, top_data, top_diff,
        len_coordinates, len_trans_params, field_channels);
    CUDA_POST_KERNEL_CHECK;
    if(transform_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      FieldProbingBackwardTrans<Dtype><<<CAFFE_GET_BLOCKS(batch_size_), CAFFE_CUDA_NUM_THREADS>>>(num_samples, num_sliding_,
          batch_size_, field_dim_, step, filters, bottom_data, trans, trans_diff, top_data, top_diff,
          len_coordinates, len_trans_params, field_channels);
      CUDA_POST_KERNEL_CHECK;
    }
  } /* field_num_ */

  caffe_gpu_scal(this->blobs_[0]->count(), Dtype(1.0 / (num_sliding_total * batch_size_ * field_num_)), filters_diff);
  if (transform_) {
    caffe_gpu_scal(bottom[field_num_]->count(), Dtype(1.0 / (slided_num_samples * field_num_)), trans_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FieldProbingLayer);

}  // namespace caffe
