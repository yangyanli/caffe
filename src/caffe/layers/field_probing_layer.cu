#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/field_probing_layer.hpp"

namespace caffe {

template<typename Dtype>
__device__ void SnapGrid_gpu(Dtype& value, int& value_0, int& value_1, const int max) {
  if (value >= 0 && value < max) {
    value_0 = floor(value);
  } else if (value < 0) {
    value = 0;
    value_0 = 0;
  } else /*(value >= max)*/ {
    value = max;
    value_0 = max-1;
  }
  value_1 = value_0 + 1;
}

template<typename Dtype>
__device__ Dtype Interpolate_gpu(const Dtype* df, const int batch_idx,
  const int x0, const int y0, const int z0,
  const int x1, const int y1, const int z1,
  const Dtype x_x0, const Dtype y_y0, const Dtype z_z0,
  const Dtype x1_x, const Dtype y1_y, const Dtype z1_z,
  const int df_dim_x, const int df_dim_y, const int df_dim_z) {
  int b_offset_000 = ((batch_idx * df_dim_x + x0) * df_dim_y + y0) * df_dim_z + z0;
  int b_offset_001 = ((batch_idx * df_dim_x + x0) * df_dim_y + y0) * df_dim_z + z1;
  int b_offset_010 = ((batch_idx * df_dim_x + x0) * df_dim_y + y1) * df_dim_z + z0;
  int b_offset_011 = ((batch_idx * df_dim_x + x0) * df_dim_y + y1) * df_dim_z + z1;
  int b_offset_100 = ((batch_idx * df_dim_x + x1) * df_dim_y + y0) * df_dim_z + z0;
  int b_offset_101 = ((batch_idx * df_dim_x + x1) * df_dim_y + y0) * df_dim_z + z1;
  int b_offset_110 = ((batch_idx * df_dim_x + x1) * df_dim_y + y1) * df_dim_z + z0;
  int b_offset_111 = ((batch_idx * df_dim_x + x1) * df_dim_y + y1) * df_dim_z + z1;

  Dtype v000 = df[b_offset_000];
  Dtype v001 = df[b_offset_001];
  Dtype v010 = df[b_offset_010];
  Dtype v011 = df[b_offset_011];
  Dtype v100 = df[b_offset_100];
  Dtype v101 = df[b_offset_101];
  Dtype v110 = df[b_offset_110];
  Dtype v111 = df[b_offset_111];

  Dtype c00 = v000*x1_x+v100*x_x0;
  Dtype c10 = v010*x1_x+v110*x_x0;
  Dtype c01 = v001*x1_x+v101*x_x0;
  Dtype c11 = v011*x1_x+v111*x_x0;

  Dtype c0 = c00*y1_y+c10*y_y0;
  Dtype c1 = c01*y1_y+c11*y_y0;

  return c0*z1_z+c1*z_z0;
}

template<typename Dtype>
__device__ void SnapGrid_gpu(Dtype& value, int& value_0, int& value_1, Dtype& value_a, Dtype& value_m, const int max) {
  SnapGrid_gpu(value, value_0, value_1, max);

  const Dtype h = 0.01;
  Dtype lower_bound = value_0+h;
  Dtype upper_bound = value_1-h;
  if (value >= lower_bound && value < upper_bound) {
    value_m = value - h;
    value_a = value + h;
  } else if (value < lower_bound) {
    value_m = value_0;
    value_a = value + h;
  } else /* (value >= upper_bound) */ {
    value_m = value - h;
    value_a = value_1;
  }
}

template<typename Dtype>
__device__ Dtype ComputeGradient_gpu(const Dtype* df, const int batch_idx,
  const int x0, const int y0, const int z0,
  const int x1, const int y1, const int z1,
  const Dtype x_a, const Dtype y_a, const Dtype z_a,
  const Dtype x_m, const Dtype y_m, const Dtype z_m,
  const Dtype x_x0, const Dtype y_y0, const Dtype z_z0,
  const Dtype x1_x, const Dtype y1_y, const Dtype z1_z,
  Dtype& dx, Dtype& dy, Dtype& dz,
  const int df_dim_x, const int df_dim_y, const int df_dim_z) {
  int b_offset_000 = ((batch_idx * df_dim_x + x0) * df_dim_y + y0) * df_dim_z + z0;
  int b_offset_001 = ((batch_idx * df_dim_x + x0) * df_dim_y + y0) * df_dim_z + z1;
  int b_offset_010 = ((batch_idx * df_dim_x + x0) * df_dim_y + y1) * df_dim_z + z0;
  int b_offset_011 = ((batch_idx * df_dim_x + x0) * df_dim_y + y1) * df_dim_z + z1;
  int b_offset_100 = ((batch_idx * df_dim_x + x1) * df_dim_y + y0) * df_dim_z + z0;
  int b_offset_101 = ((batch_idx * df_dim_x + x1) * df_dim_y + y0) * df_dim_z + z1;
  int b_offset_110 = ((batch_idx * df_dim_x + x1) * df_dim_y + y1) * df_dim_z + z0;
  int b_offset_111 = ((batch_idx * df_dim_x + x1) * df_dim_y + y1) * df_dim_z + z1;
              
  Dtype v000 = df[b_offset_000];
  Dtype v001 = df[b_offset_001];
  Dtype v010 = df[b_offset_010];
  Dtype v011 = df[b_offset_011];
  Dtype v100 = df[b_offset_100];
  Dtype v101 = df[b_offset_101];
  Dtype v110 = df[b_offset_110];
  Dtype v111 = df[b_offset_111];

  Dtype x_am = x_a-x_m;
  Dtype x_ma = x_m-x_a;
  Dtype y_am = y_a-y_m;
  Dtype y_ma = y_m-y_a;
  Dtype z_am = z_a-z_m;
  Dtype z_ma = z_m-z_a;

  dx =
  v000*x_ma*y1_y*z1_z +
  v100*x_am*y1_y*z1_z +
  v010*x_ma*y_y0*z1_z + 
  v001*x_ma*y1_y*z_z0 +
  v101*x_am*y1_y*z_z0 +
  v011*x_ma*y_y0*z_z0 +
  v110*x_am*y_y0*z1_z +
  v111*x_am*y_y0*z_z0;
  dx /= x_am;
  
  dy = 
  v000*x1_x*y_ma*z1_z +
  v100*x_x0*y_ma*z1_z +
  v010*x1_x*y_am*z1_z + 
  v001*x1_x*y_ma*z_z0 +
  v101*x_x0*y_ma*z_z0 +
  v011*x1_x*y_am*z_z0 +
  v110*x_x0*y_am*z1_z +
  v111*x_x0*y_am*z_z0;
  dy /= y_am;
  
  dz = 
  v000*x1_x*y1_y*z_ma +
  v100*x_x0*y1_y*z_ma +
  v010*x1_x*y_y0*z_ma + 
  v001*x1_x*y1_y*z_am +
  v101*x_x0*y1_y*z_am +
  v011*x1_x*y_y0*z_am +
  v110*x_x0*y_y0*z_ma +
  v111*x_x0*y_y0*z_am;
  dz /= z_am;

  return (
  v000*x1_x*y1_y*z1_z +
  v100*x_x0*y1_y*z1_z +
  v010*x1_x*y_y0*z1_z +
  v001*x1_x*y1_y*z_z0 +
  v101*x_x0*y1_y*z_z0 +
  v011*x1_x*y_y0*z_z0 +
  v110*x_x0*y_y0*z1_z +
  v111*x_x0*y_y0*z_z0);
}

template<typename Dtype>
__device__ Dtype ComputeGradient_gpu(const Dtype* df, const int batch_idx,
  Dtype& x, Dtype& y, Dtype& z,
  Dtype& dx, Dtype& dy, Dtype& dz,
  const int df_dim_x, const int df_dim_y, const int df_dim_z) {
  int x0, y0, z0, x1, y1, z1;
  Dtype x_a, y_a, z_a, x_m, y_m, z_m;
  SnapGrid_gpu(x, x0, x1, x_a, x_m, df_dim_x-1);
  SnapGrid_gpu(y, y0, y1, y_a, y_m, df_dim_y-1);
  SnapGrid_gpu(z, z0, z1, z_a, z_m, df_dim_z-1);
  Dtype x_x0 = x-x0;
  Dtype y_y0 = y-y0;
  Dtype z_z0 = z-z0;
  Dtype x1_x = x1-x;
  Dtype y1_y = y1-y;
  Dtype z1_z = z1-z;

  return ComputeGradient_gpu(df, batch_idx, x0, y0, z0, x1, y1, z1, x_a, y_a, z_a, x_m, y_m, z_m,
                  x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, dx, dy, dz, df_dim_x, df_dim_y, df_dim_z);
}

template<typename Dtype>
__device__ void Jitter_gpu(const Dtype& value, Dtype& value_a, Dtype& value_m, const int max) {
  const Dtype h = 0.01;
  if (value >= h && value < max-h) {
    value_m = value - h;
    value_a = value + h;
  } else if (value < h) {
    value_m = 0;
    value_a = value + h;
  } else /* (value >= max-h) */ {
    value_m = value - h;
    value_a = max;
  }
}

template<typename Dtype>
__device__ void Normalize_gpu(Dtype& nx, Dtype& ny, Dtype& nz) {
  Dtype len = sqrt(nx*nx+ny*ny+nz*nz);
  if (len != 0) {
    nx /= len;
    ny /= len;
    nz /= len;
  }
}

template<typename Dtype>
__device__ void ComputeNormalGradient_gpu(const Dtype* df, const int batch_idx,
  Dtype& x, Dtype& y, Dtype& z,
  Dtype& nx_dx, Dtype& nx_dy, Dtype& nx_dz,
  Dtype& ny_dx, Dtype& ny_dy, Dtype& ny_dz,
  Dtype& nz_dx, Dtype& nz_dy, Dtype& nz_dz,
  const int df_dim_x, const int df_dim_y, const int df_dim_z) {
  Dtype x_a, y_a, z_a, x_m, y_m, z_m;
  Jitter_gpu(x, x_a, x_m, df_dim_x-1);
  Jitter_gpu(y, y_a, y_m, df_dim_y-1);
  Jitter_gpu(z, z_a, z_m, df_dim_z-1);

  Dtype nx_x_a, ny_x_a, nz_x_a;
  ComputeGradient_gpu(df, batch_idx, x_a, y, z, nx_x_a, ny_x_a, nz_x_a, df_dim_x, df_dim_y, df_dim_z);
  Normalize_gpu(nx_x_a, ny_x_a, nz_x_a);

  Dtype nx_x_m, ny_x_m, nz_x_m;
  ComputeGradient_gpu(df, batch_idx, x_m, y, z, nx_x_m, ny_x_m, nz_x_m, df_dim_x, df_dim_y, df_dim_z);
  Normalize_gpu(nx_x_m, ny_x_m, nz_x_m);

  Dtype nx_y_a, ny_y_a, nz_y_a;
  ComputeGradient_gpu(df, batch_idx, x, y_a, z, nx_y_a, ny_y_a, nz_y_a, df_dim_x, df_dim_y, df_dim_z);
  Normalize_gpu(nx_y_a, ny_y_a, nz_y_a);

  Dtype nx_y_m, ny_y_m, nz_y_m;
  ComputeGradient_gpu(df, batch_idx, x, y_m, z, nx_y_m, ny_y_m, nz_y_m, df_dim_x, df_dim_y, df_dim_z);
  Normalize_gpu(nx_y_m, ny_y_m, nz_y_m);

  Dtype nx_z_a, ny_z_a, nz_z_a;
  ComputeGradient_gpu(df, batch_idx, x, y, z_a, nx_z_a, ny_z_a, nz_z_a, df_dim_x, df_dim_y, df_dim_z);
  Normalize_gpu(nx_z_a, ny_z_a, nz_z_a);

  Dtype nx_z_m, ny_z_m, nz_z_m;
  ComputeGradient_gpu(df, batch_idx, x, y, z_m, nx_z_m, ny_z_m, nz_z_m, df_dim_x, df_dim_y, df_dim_z);
  Normalize_gpu(nx_z_m, ny_z_m, nz_z_m);

  Dtype x_scaler = 1.0/(x_a-x_m);
  Dtype y_scaler = 1.0/(y_a-y_m);
  Dtype z_scaler = 1.0/(z_a-z_m);

  nx_dx = (nx_x_a-nx_x_m)*x_scaler;
  nx_dy = (nx_y_a-nx_y_m)*y_scaler;
  nx_dz = (nx_z_a-nx_z_m)*z_scaler;

  ny_dx = (ny_x_a-ny_x_m)*x_scaler;
  ny_dy = (ny_y_a-ny_y_m)*y_scaler;
  ny_dz = (ny_z_a-ny_z_m)*z_scaler;

  nz_dx = (nz_x_a-nz_x_m)*x_scaler;
  nz_dy = (nz_y_a-nz_y_m)*y_scaler;
  nz_dz = (nz_z_a-nz_z_m)*z_scaler;
}

template <typename Dtype>
__global__ void FieldProbingForward(const int num_samples, const int batch_size,
    const int df_dim_x, const int df_dim_y, const int df_dim_z, const int df_dim_x_1, const int df_dim_y_1, const int df_dim_z_1,
    const Dtype* weight, const Dtype* bottom_data, Dtype* top_data) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each sample
  if(sample_idx < num_samples) {
    int w_offset = sample_idx * 3;
    Dtype x = weight[w_offset+0];
    Dtype y = weight[w_offset+1];
    Dtype z = weight[w_offset+2];

    int x0, y0, z0, x1, y1, z1;
    SnapGrid_gpu(x, x0, x1, df_dim_x_1);
    SnapGrid_gpu(y, y0, y1, df_dim_y_1);
    SnapGrid_gpu(z, z0, z1, df_dim_z_1);
    Dtype x_x0 = x-x0;
    Dtype y_y0 = y-y0;
    Dtype z_z0 = z-z0;
    Dtype x1_x = x1-x;
    Dtype y1_y = y1-y;
    Dtype z1_z = z1-z;
    int top_count = num_samples*batch_size;
    for (int top_offset = sample_idx, batch_idx = 0; top_offset < top_count; top_offset += num_samples) {
      top_data[top_offset] = Interpolate_gpu(bottom_data, batch_idx, x0, y0, z0, x1, y1, z1, x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, df_dim_x, df_dim_y, df_dim_z);
      batch_idx ++;
    }
  }
}

template <typename Dtype>
__global__ void FieldProbingForwardWithNormal(const int num_samples, const int batch_size,
    const int df_dim_x, const int df_dim_y, const int df_dim_z, const int df_dim_x_1, const int df_dim_y_1, const int df_dim_z_1,
    const Dtype* weight, const Dtype* bottom_data, Dtype* top_data, Dtype* top_normal_data) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each sample
  if(sample_idx < num_samples) {
    int w_offset = sample_idx * 3;
    Dtype x = weight[w_offset+0];
    Dtype y = weight[w_offset+1];
    Dtype z = weight[w_offset+2];

    int x0, y0, z0, x1, y1, z1;
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    SnapGrid_gpu(x, x0, x1, x_a, x_m, df_dim_x_1);
    SnapGrid_gpu(y, y0, y1, y_a, y_m, df_dim_y_1);
    SnapGrid_gpu(z, z0, z1, z_a, z_m, df_dim_z_1);
    Dtype x_x0 = x-x0;
    Dtype y_y0 = y-y0;
    Dtype z_z0 = z-z0;
    Dtype x1_x = x1-x;
    Dtype y1_y = y1-y;
    Dtype z1_z = z1-z;

    Dtype nx, ny, nz;
    int top_count = num_samples*batch_size;
    for (int top_offset = sample_idx, batch_idx = 0; top_offset < top_count; top_offset += num_samples) {
      top_data[top_offset] = ComputeGradient_gpu(bottom_data, batch_idx, x0, y0, z0, x1, y1, z1, x_a, y_a, z_a, x_m, y_m, z_m,
                      x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, nx, ny, nz, df_dim_x, df_dim_y, df_dim_z);
      Normalize_gpu(nx, ny, nz);
      top_normal_data[top_offset*3+0] = nx;
      top_normal_data[top_offset*3+1] = ny;
      top_normal_data[top_offset*3+2] = nz;

      batch_idx ++;
    }
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();

  const vector<int>& bottom_shape = bottom[0]->shape();
  int batch_size = bottom_shape[0];
  int df_dim_x = bottom_shape[1];
  int df_dim_y = bottom_shape[2];
  int df_dim_z = bottom_shape[3];
  int df_dim_x_1 = df_dim_x-1;
  int df_dim_y_1 = df_dim_y-1;
  int df_dim_z_1 = df_dim_z-1;
  int num_grids = dim_grid_*dim_grid_*dim_grid_;
  int num_samples = num_grids*num_curve_*len_curve_;

  for (int i = 0; i < bottom.size(); ++ i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    if(output_normal_) {
      Dtype* top_data = top[2*i+0]->mutable_gpu_data();
      Dtype* top_normal_data = top[2*i+1]->mutable_gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      FieldProbingForwardWithNormal<Dtype><<<CAFFE_GET_BLOCKS(num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, batch_size,
        df_dim_x, df_dim_y, df_dim_z, df_dim_x_1, df_dim_y_1, df_dim_z_1,
        weight, bottom_data, top_data, top_normal_data);
      CUDA_POST_KERNEL_CHECK;
    } else {
      Dtype* top_data = top[i]->mutable_gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      FieldProbingForward<Dtype><<<CAFFE_GET_BLOCKS(num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, batch_size,
        df_dim_x, df_dim_y, df_dim_z, df_dim_x_1, df_dim_y_1, df_dim_z_1,
        weight, bottom_data, top_data);
      CUDA_POST_KERNEL_CHECK;
    }
  } /* bottom.size() */
}

template <typename Dtype>
__global__ void FieldProbingBackward(const int num_samples, const int batch_size,
    const int df_dim_x, const int df_dim_y, const int df_dim_z, const int df_dim_x_1, const int df_dim_y_1, const int df_dim_z_1,
    const Dtype* weight, const Dtype* bottom_data, const Dtype* top_diff, Dtype* weight_diff) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;

  // One thread for each sample
  if(sample_idx < num_samples) {
    int w_offset = sample_idx * 3;
    Dtype x = weight[w_offset+0];
    Dtype y = weight[w_offset+1];
    Dtype z = weight[w_offset+2];
  
    int x0, y0, z0, x1, y1, z1;
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    SnapGrid_gpu(x, x0, x1, x_a, x_m, df_dim_x_1);
    SnapGrid_gpu(y, y0, y1, y_a, y_m, df_dim_y_1);
    SnapGrid_gpu(z, z0, z1, z_a, z_m, df_dim_z_1);
    Dtype x_x0 = x-x0;
    Dtype y_y0 = y-y0;
    Dtype z_z0 = z-z0;
    Dtype x1_x = x1-x;
    Dtype y1_y = y1-y;
    Dtype z1_z = z1-z;
  
    Dtype w_diff_x = 0;
    Dtype w_diff_y = 0;
    Dtype w_diff_z = 0;
    Dtype dx, dy, dz;
    int top_count = num_samples*batch_size;
    for (int top_offset = sample_idx, batch_idx = 0; top_offset < top_count; top_offset += num_samples) {
    ComputeGradient_gpu(bottom_data, batch_idx, x0, y0, z0, x1, y1, z1, x_a, y_a, z_a, x_m, y_m, z_m,
      x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, dx, dy, dz, df_dim_x, df_dim_y, df_dim_z);
      const Dtype& t_diff = top_diff[top_offset];
      w_diff_x += t_diff*dx;
      w_diff_y += t_diff*dy;
      w_diff_z += t_diff*dz;

      batch_idx ++;
    }
    weight_diff[w_offset+0] += w_diff_x;
    weight_diff[w_offset+1] += w_diff_y;
    weight_diff[w_offset+2] += w_diff_z;
  }
}

template <typename Dtype>
__global__ void FieldProbingBackwardWithNormal(const int num_samples, const int batch_size,
    const int df_dim_x, const int df_dim_y, const int df_dim_z, const int df_dim_x_1, const int df_dim_y_1, const int df_dim_z_1,
    const Dtype* weight, const Dtype* bottom_data, const Dtype* top_diff, const Dtype* top_normal_diff, Dtype* weight_diff) {
  int sample_idx = blockDim.x*blockIdx.x + threadIdx.x;

  // One thread for each sample
  if(sample_idx < num_samples) {
    int w_offset = sample_idx * 3;
    Dtype x = weight[w_offset+0];
    Dtype y = weight[w_offset+1];
    Dtype z = weight[w_offset+2];

    int x0, y0, z0, x1, y1, z1;
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    SnapGrid_gpu(x, x0, x1, x_a, x_m, df_dim_x_1);
    SnapGrid_gpu(y, y0, y1, y_a, y_m, df_dim_y_1);
    SnapGrid_gpu(z, z0, z1, z_a, z_m, df_dim_z_1);
    Dtype x_x0 = x-x0;
    Dtype y_y0 = y-y0;
    Dtype z_z0 = z-z0;
    Dtype x1_x = x1-x;
    Dtype y1_y = y1-y;
    Dtype z1_z = z1-z;

    Dtype w_diff_x = 0;
    Dtype w_diff_y = 0;
    Dtype w_diff_z = 0;
    Dtype dx, dy, dz;
    Dtype nx_dx, nx_dy, nx_dz;
    Dtype ny_dx, ny_dy, ny_dz;
    Dtype nz_dx, nz_dy, nz_dz;
    int top_count = num_samples*batch_size;
    for (int top_offset = sample_idx, batch_idx = 0; top_offset < top_count; top_offset += num_samples) {
      ComputeGradient_gpu(bottom_data, batch_idx,
          x0, y0, z0, x1, y1, z1,
          x_a, y_a, z_a, x_m, y_m, z_m,
          x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
          dx, dy, dz, df_dim_x, df_dim_y, df_dim_z);
      const Dtype& t_diff = top_diff[top_offset];
      w_diff_x += t_diff*dx;
      w_diff_y += t_diff*dy;
      w_diff_z += t_diff*dz;

      ComputeNormalGradient_gpu(bottom_data, batch_idx,
          x, y, z,
          nx_dx, nx_dy, nx_dz,
          ny_dx, ny_dy, ny_dz,
          nz_dx, nz_dy, nz_dz,
          df_dim_x, df_dim_y, df_dim_z);
      const Dtype& tx_diff = top_normal_diff[3*top_offset+0];
      const Dtype& ty_diff = top_normal_diff[3*top_offset+1];
      const Dtype& tz_diff = top_normal_diff[3*top_offset+2];

      w_diff_x += tx_diff*nx_dx + ty_diff*ny_dx + tz_diff*nz_dx;
      w_diff_y += tx_diff*nx_dy + ty_diff*ny_dy + tz_diff*nz_dy;
      w_diff_z += tx_diff*nx_dz + ty_diff*ny_dz + tz_diff*nz_dz;

      batch_idx ++;
    }
    weight_diff[w_offset+0] += w_diff_x;
    weight_diff[w_offset+1] += w_diff_y;
    weight_diff[w_offset+2] += w_diff_z;
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const vector<int>& bottom_shape = bottom[0]->shape();
  int batch_size = bottom_shape[0];
  int df_dim_x = bottom_shape[1];
  int df_dim_y = bottom_shape[2];
  int df_dim_z = bottom_shape[3];
  int df_dim_x_1 = df_dim_x-1;
  int df_dim_y_1 = df_dim_y-1;
  int df_dim_z_1 = df_dim_z-1;
  int num_grids = dim_grid_*dim_grid_*dim_grid_;
  int num_samples = num_grids*num_curve_*len_curve_;

  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* top_diff = NULL;
    const Dtype* top_normal_diff = NULL;
    if (output_normal_) {
      top_diff = top[2*i+0]->gpu_diff();
      top_normal_diff = top[2*i+1]->gpu_diff();
    } else {
      top_diff = top[i]->gpu_diff();
    }
    if (this->param_propagate_down_[0]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      if(output_normal_) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        FieldProbingBackwardWithNormal<Dtype><<<CAFFE_GET_BLOCKS(num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, batch_size,
          df_dim_x, df_dim_y, df_dim_z, df_dim_x_1, df_dim_y_1, df_dim_z_1,
          weight, bottom_data, top_diff, top_normal_diff, weight_diff);
        CUDA_POST_KERNEL_CHECK;
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        FieldProbingBackward<Dtype><<<CAFFE_GET_BLOCKS(num_samples), CAFFE_CUDA_NUM_THREADS>>>(num_samples, batch_size,
          df_dim_x, df_dim_y, df_dim_z, df_dim_x_1, df_dim_y_1, df_dim_z_1,
          weight, bottom_data, top_diff, weight_diff);
        CUDA_POST_KERNEL_CHECK;
      }
    }
    
    if (rand()%100 == 0) {
      Dtype amax, aavg;
      caffe_gpu_amax(top[i]->count(), top_diff, &amax);
      caffe_gpu_aavg(top[i]->count(), top_diff, &aavg);
      LOG(INFO) << "FieldProbingLayer::Backward_gpu top_diff max-avg: " << amax << "\t" << aavg;
    }
  } /* top.size() */
 
  if (rand()%100 == 0) {
    Dtype amax, aavg;
    caffe_gpu_amax(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), &amax);
    caffe_gpu_aavg(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), &aavg);
    LOG(INFO) << "FieldProbingLayer::Backward_gpu weight_diff max-avg: " << amax << "\t" << aavg;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FieldProbingLayer);

}  // namespace caffe
