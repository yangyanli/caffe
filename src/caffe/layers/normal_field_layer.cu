#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/normal_field_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void NormalFieldForward(const int num_grids, const int grid_dim, const int batch_size,
  const Dtype* bottom_data, Dtype* top_data) {
  const int grid_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each grid
  if(grid_idx < num_grids) {
    const int grid_dim_1 = grid_dim-1;
    const int yz = grid_dim*grid_dim;
    Dtype z = grid_idx%grid_dim + 0.5;
    Dtype y = (grid_idx/grid_dim)%grid_dim + 0.5;
    Dtype x = grid_idx/yz + 0.5;
    int x0, y0, z0, x1, y1, z1;
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    SnapGrid_gpu(x, x0, x1, x_a, x_m, grid_dim_1);
    SnapGrid_gpu(y, y0, y1, y_a, y_m, grid_dim_1);
    SnapGrid_gpu(z, z0, z1, z_a, z_m, grid_dim_1);
    Dtype x_x0 = x-x0;
    Dtype y_y0 = y-y0;
    Dtype z_z0 = z-z0;
    Dtype x1_x = x1-x;
    Dtype y1_y = y1-y;
    Dtype z1_z = z1-z;
    Dtype nx, ny, nz;
    for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
      ComputeGradient_gpu(bottom_data, batch_idx, x0, y0, z0, x1, y1, z1,
        x_a, y_a, z_a, x_m, y_m, z_m, x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
        nx, ny, nz, grid_dim, grid_dim, grid_dim);
      Normalize_gpu(nx, ny, nz);
      int p = (batch_idx*num_grids+grid_idx)*3;
      top_data[p+0] = nx;
      top_data[p+1] = ny;
      top_data[p+2] = nz;
    }
  }
}

template<typename Dtype>
void NormalFieldLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const vector<int>& bottom_shape = bottom[0]->shape();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int batch_size = bottom_shape[0];
  const int grid_dim = bottom_shape[1];
  const int num_grids = bottom[0]->count(1);

  // NOLINT_NEXT_LINE(whitespace/operators)
  NormalFieldForward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(num_grids, grid_dim, batch_size, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalFieldLayer);

}  // namespace caffe
