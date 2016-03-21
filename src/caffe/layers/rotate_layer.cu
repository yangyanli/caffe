#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/rotate_layer.hpp"

namespace caffe {

template<typename Dtype>
__device__ void SnapGrid_gpu(Dtype& value, int& value_0, int& value_1, const int max) {
  if (value > 0 && value < max) {
    value_0 = floor(value);
  } else if (value <= 0) {
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


template <typename Dtype>
__global__ void RotateForward(const int num_grids, const int grid_dim, const int batch_size, const int num_rotation,
    const Dtype pad_value, const Dtype* bottom_data, const Dtype* rotations, Dtype* top_data) {
  const int t_grid_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each grid
  if(t_grid_idx < num_grids) {
    Dtype c_offset = (grid_dim-1)/2.0;
    int grid_dim_1 = grid_dim-1;
    const int yz = grid_dim*grid_dim;
    for (int b_batch_idx = 0; b_batch_idx < batch_size; ++ b_batch_idx) {
      int offset = b_batch_idx * num_rotation;
      for(int rotation_idx = 0; rotation_idx < num_rotation; ++ rotation_idx) {
        int t_batch_idx = offset + rotation_idx;

        int r_offset = t_batch_idx*9;
        Dtype r00 = rotations[r_offset++];
        Dtype r01 = rotations[r_offset++];
        Dtype r02 = rotations[r_offset++];
        Dtype r10 = rotations[r_offset++];
        Dtype r11 = rotations[r_offset++];
        Dtype r12 = rotations[r_offset++];
        Dtype r20 = rotations[r_offset++];
        Dtype r21 = rotations[r_offset++];
        Dtype r22 = rotations[r_offset++];

        int tz = t_grid_idx%grid_dim;
        int ty = (t_grid_idx/grid_dim)%grid_dim;
        int tx = t_grid_idx/yz;

        Dtype txx = tx+0.5-c_offset;
        Dtype tyy = ty+0.5-c_offset;
        Dtype tzz = tz+0.5-c_offset;

        Dtype bx = r00*txx + r01*tyy + r02*tzz + c_offset;
        Dtype by = r10*txx + r11*tyy + r12*tzz + c_offset;
        Dtype bz = r20*txx + r21*tyy + r22*tzz + c_offset;

        if(bx >= 0 && bx < grid_dim
            && by >= 0 && by < grid_dim
            && bz >= 0 && bz < grid_dim) {
          int x0, y0, z0, x1, y1, z1;
          SnapGrid_gpu(bx, x0, x1, grid_dim_1);
          SnapGrid_gpu(by, y0, y1, grid_dim_1);
          SnapGrid_gpu(bz, z0, z1, grid_dim_1);
          Dtype x_x0 = bx-x0;
          Dtype y_y0 = by-y0;
          Dtype z_z0 = bz-z0;
          Dtype x1_x = x1-bx;
          Dtype y1_y = y1-by;
          Dtype z1_z = z1-bz;
          top_data[t_batch_idx*num_grids+t_grid_idx] = Interpolate_gpu(bottom_data, b_batch_idx, x0, y0, z0, x1, y1, z1, x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, grid_dim, grid_dim, grid_dim);
        } else {
          top_data[t_batch_idx*num_grids+t_grid_idx] = pad_value;
        }
      } /* rotation_idx */
    } /* b_batch_idx */
  }
}

template <typename Dtype>
void RotateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ForwardLabel(bottom[1], top[1]);

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const vector<int>& bottom_shape = bottom[0]->shape();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int batch_size = bottom_shape[0];
  const int grid_dim = bottom_shape[1];
  const int num_grids = bottom[0]->count(1);
  const Dtype* rotations_data = rotations_.gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  RotateForward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(num_grids, grid_dim, batch_size, num_rotation_,
      pad_value_, bottom_data, rotations_data, top_data);
  CUDA_POST_KERNEL_CHECK;

  //Dtype amax, aavg;
  //caffe_gpu_amax(top[0]->count(), top[0]->gpu_data(), &amax);
  //caffe_gpu_aavg(top[0]->count(), top[0]->gpu_data(), &aavg);
  //LOG(INFO) << "RotateLayer::Forward_gpu top_data max-avg: " << amax << "\t" << aavg;
}

INSTANTIATE_LAYER_GPU_FUNCS(RotateLayer);


}  // namespace caffe
