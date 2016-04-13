#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/gradient_field_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GradientFieldForward(const int num_grids, const int grid_dim, const int batch_size,
  const Dtype* bottom_data, Dtype* top_data, const int field_channels) {
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
    for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
      Dtype* t_data = top_data + (batch_idx*num_grids+grid_idx)*field_channels*3;
      ComputeGradient_gpu(bottom_data, batch_idx, x0, y0, z0, x1, y1, z1,
        x_a, y_a, z_a, x_m, y_m, z_m, x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
        grid_dim, grid_dim, grid_dim, t_data, field_channels);
      Normalize_gpu(t_data);
    }
  }
}

template<typename Dtype>
void GradientFieldLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const vector<int>& field_shape = bottom[0]->shape();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int batch_size = field_shape[0];
  const int grid_dim = field_shape[1];
  const int num_grids = grid_dim*grid_dim*grid_dim;
  int field_channels = (field_shape.size() == 5)?(field_shape.back()):(1);

  // NOLINT_NEXT_LINE(whitespace/operators)
  GradientFieldForward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(num_grids, grid_dim, batch_size, bottom_data, top_data, field_channels);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(GradientFieldLayer);

}  // namespace caffe
