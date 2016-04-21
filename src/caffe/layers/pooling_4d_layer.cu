#include <cfloat>

#include "caffe/layers/pooling_4d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int num_grids, const Dtype* bottom_data, const int batch_size,
    const int spatial_, const int channel_, const int pooled_spatial_, const int pooled_channel_,
    const int kernel_s_, const int kernel_c_, const int stride_s_, const int stride_c_, const int pad_s_, const int pad_c_,
    Dtype* top_data, unsigned int* mask) {
  const int grid_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each grid
  if(grid_idx < num_grids) {
    int idx = grid_idx;
    int c = idx % pooled_channel_;
    idx /= pooled_channel_;
    int z = idx % pooled_spatial_;
    idx /= pooled_spatial_;
    int y = idx % pooled_spatial_;
    int x = idx / pooled_spatial_;

    int xs = x*stride_s_-pad_s_;
    int xe = min(xs+kernel_s_, spatial_);
    xs = max(xs, 0);
    int ys = y*stride_s_-pad_s_;
    int ye = min(ys+kernel_s_, spatial_);
    ys = max(ys, 0);
    int zs = z*stride_s_-pad_s_;
    int ze = min(zs+kernel_s_, spatial_);
    zs = max(zs, 0);
    int cs = c*stride_c_-pad_c_;
    int ce = min(cs+kernel_c_, channel_);
    cs = max(cs, 0);

    int sc = spatial_*channel_;
    int ssc = spatial_*sc;
    int sssc = spatial_*ssc;
    int top_count = num_grids*batch_size;
    for (int top_offset = grid_idx, batch_idx = 0; top_offset < top_count; top_offset += num_grids) {
      int max_idx = 0;
      Dtype max_value = -FLT_MAX;
      int n_offset = batch_idx * sssc;
      for (int xx = xs; xx < xe; ++ xx) {
        int n_x_offset = n_offset + xx*ssc; 
        for (int yy = ys; yy < ye; ++ yy) {
          int n_x_y_offset = n_x_offset + yy*sc;
          for (int zz = zs; zz < ze; ++ zz) {
            int n_x_y_z_offset = n_x_y_offset + zz*channel_;
            for (int cc = cs; cc < ce; ++ cc) {
              int bottom_offset = n_x_y_z_offset+cc;
              const Dtype& bottom_value = bottom_data[bottom_offset];
              if (bottom_value > max_value) {
                max_value = bottom_value;
                max_idx = bottom_offset;
              }
            }
          }
        }
      }
      mask[top_offset] = max_idx;
      top_data[top_offset] = max_value;

      batch_idx ++;
    }
  }
}


template <typename Dtype>
__global__ void AvgPoolForward(const int num_grids, const Dtype* bottom_data, const int batch_size,
    const int spatial_, const int channel_, const int pooled_spatial_, const int pooled_channel_,
    const int kernel_s_, const int kernel_c_, const int stride_s_, const int stride_c_, const int pad_s_, const int pad_c_,
    Dtype* top_data) {
  const int grid_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each grid
  if(grid_idx < num_grids) {
    int idx = grid_idx;
    int c = idx % pooled_channel_;
    idx /= pooled_channel_;
    int z = idx % pooled_spatial_;
    idx /= pooled_spatial_;
    int y = idx % pooled_spatial_;
    int x = idx / pooled_spatial_;

    int xs = x*stride_s_-pad_s_;
    int xe = min(xs+kernel_s_, spatial_+pad_s_);
    int pool_size_x = xe-xs;
    xs = max(xs, 0);
    xe = min(xe, spatial_);
    int ys = y*stride_s_-pad_s_;
    int ye = min(ys+kernel_s_, spatial_+pad_s_);
    int pool_size_y = ye-ys;
    ys = max(ys, 0);
    ye = min(ye, spatial_);
    int zs = z*stride_s_-pad_s_;
    int ze = min(zs+kernel_s_, spatial_+pad_s_);
    int pool_size_z = ze-zs;
    zs = max(zs, 0);
    ze = min(ze, spatial_);
    int cs = c*stride_c_-pad_c_;
    int ce = min(cs+kernel_c_, channel_+pad_c_);
    int pool_size_c = ce-cs;
    cs = max(cs, 0);
    ce = min(ce, channel_);
    Dtype pool_size = pool_size_x*pool_size_y*pool_size_z*pool_size_c;

    int sc = spatial_*channel_;
    int ssc = spatial_*sc;
    int sssc = spatial_*ssc;
    int top_count = num_grids*batch_size;
    for (int top_offset = grid_idx, batch_idx = 0; top_offset < top_count; top_offset += num_grids) {
      Dtype avg_value = 0;
      int n_offset = batch_idx * sssc;
      for (int xx = xs; xx < xe; ++ xx) {
        int n_x_offset = n_offset + xx*ssc; 
        for (int yy = ys; yy < ye; ++ yy) {
          int n_x_y_offset = n_x_offset + yy*sc;
          for (int zz = zs; zz < ze; ++ zz) {
            int n_x_y_z_offset = n_x_y_offset + zz*channel_;
            for (int cc = cs; cc < ce; ++ cc) {
              int bottom_offset = n_x_y_z_offset+cc;
              avg_value += bottom_data[bottom_offset];
            }
          }
        }
      }
      top_data[top_offset] = avg_value/pool_size;

      batch_idx ++;
    }
  }
}

template <typename Dtype>
void Pooling4DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int batch_size = bottom[0]->shape(0);
  const int num_grids = pooled_spatial_*pooled_spatial_*pooled_spatial_*pooled_channel_;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(num_grids, bottom_data, batch_size,
      spatial_, channel_, pooled_spatial_, pooled_channel_,
      kernel_s_, kernel_c_, stride_s_, stride_c_, pad_s_, pad_c_, top_data, max_idx_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvgPoolForward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(num_grids, bottom_data, batch_size,
      spatial_, channel_, pooled_spatial_, pooled_channel_,
      kernel_s_, kernel_c_, stride_s_, stride_c_, pad_s_, pad_c_, top_data);
    CUDA_POST_KERNEL_CHECK;
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int num_grids, const Dtype* top_diff, const unsigned int* mask, const int batch_size,
    Dtype* bottom_diff) {
  const int grid_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each grid
  if(grid_idx < num_grids) {
    int top_count = num_grids*batch_size;
    for (int top_offset = grid_idx; top_offset < top_count; top_offset += num_grids) {
      bottom_diff[mask[top_offset]] += top_diff[top_offset];
    }
  }
}


template <typename Dtype>
__global__ void AvgPoolBackward(const int num_grids, const Dtype* top_diff, const int batch_size,
    const int spatial_, const int channel_, const int pooled_spatial_, const int pooled_channel_,
    const int kernel_s_, const int kernel_c_, const int stride_s_, const int stride_c_, const int pad_s_, const int pad_c_,
    Dtype* bottom_diff) {
  const int grid_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each grid
  if(grid_idx < num_grids) {
    int idx = grid_idx;
    int c = idx % pooled_channel_;
    idx /= pooled_channel_;
    int z = idx % pooled_spatial_;
    idx /= pooled_spatial_;
    int y = idx % pooled_spatial_;
    int x = idx / pooled_spatial_;

    int xs = x*stride_s_-pad_s_;
    int xe = min(xs+kernel_s_, spatial_+pad_s_);
    int pool_size_x = xe-xs;
    xs = max(xs, 0);
    xe = min(xe, spatial_);
    int ys = y*stride_s_-pad_s_;
    int ye = min(ys+kernel_s_, spatial_+pad_s_);
    int pool_size_y = ye-ys;
    ys = max(ys, 0);
    ye = min(ye, spatial_);
    int zs = z*stride_s_-pad_s_;
    int ze = min(zs+kernel_s_, spatial_+pad_s_);
    int pool_size_z = ze-zs;
    zs = max(zs, 0);
    ze = min(ze, spatial_);
    int cs = c*stride_c_-pad_c_;
    int ce = min(cs+kernel_c_, channel_+pad_c_);
    int pool_size_c = ce-cs;
    cs = max(cs, 0);
    ce = min(ce, spatial_);
    Dtype pool_size = pool_size_x*pool_size_y*pool_size_z*pool_size_c;

    int sc = spatial_*channel_;
    int ssc = spatial_*sc;
    int sssc = spatial_*ssc;
    int top_count = num_grids*batch_size;
    for (int top_offset = grid_idx, batch_idx = 0; top_offset < top_count; top_offset += num_grids) {
      Dtype diff = top_diff[top_offset]/pool_size;
      int n_offset = batch_idx * sssc;
      for (int xx = xs; xx < xe; ++ xx) {
        int n_x_offset = n_offset + xx*ssc; 
        for (int yy = ys; yy < ye; ++ yy) {
          int n_x_y_offset = n_x_offset + yy*sc;
          for (int zz = zs; zz < ze; ++ zz) {
            int n_x_y_z_offset = n_x_y_offset + zz*channel_;
            for (int cc = cs; cc < ce; ++ cc) {
              int bottom_offset = n_x_y_z_offset+cc;
              bottom_diff[bottom_offset] += diff;
            }
          }
        }
      }

      batch_idx ++;
    }
  }
}

template <typename Dtype>
void Pooling4DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  const int batch_size = bottom[0]->shape(0);
  const int num_grids = pooled_spatial_*pooled_spatial_*pooled_spatial_*pooled_channel_;
 
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(num_grids, top_diff, max_idx_.gpu_data(), batch_size,
      bottom_diff);
    CUDA_POST_KERNEL_CHECK;
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvgPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(num_grids, top_diff, batch_size,
      spatial_, channel_, pooled_spatial_, pooled_channel_,
      kernel_s_, kernel_c_, stride_s_, stride_c_, pad_s_, pad_c_, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(Pooling4DLayer);


}  // namespace caffe
