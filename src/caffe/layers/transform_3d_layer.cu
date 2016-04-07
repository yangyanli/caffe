#include "caffe/util/field_operations.hpp"

#include "caffe/layers/transform_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Transform3DForward(const int num_grids, const int grid_dim, const int batch_size, const int num_transformations,
    const Dtype pad_value, const Dtype* bottom_data, const Dtype* transformations, Dtype* top_data, const int len_transformation_param) {
  const int t_grid_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each grid
  if(t_grid_idx < num_grids) {
    Dtype c_offset = (grid_dim-1)/2.0;
    int grid_dim_1 = grid_dim-1;
    const int yz = grid_dim*grid_dim;
    for (int b_batch_idx = 0; b_batch_idx < batch_size; ++ b_batch_idx) {
      int offset = b_batch_idx * num_transformations;
      for(int transformation_idx = 0; transformation_idx < num_transformations; ++ transformation_idx) {
        int t_batch_idx = offset + transformation_idx;

        int p = t_batch_idx*len_transformation_param;
        Dtype a = transformations[p++];
        Dtype b = transformations[p++];
        Dtype c = transformations[p++];
        Dtype tx = transformations[p++];
        Dtype d = transformations[p++];
        Dtype e = transformations[p++];
        Dtype f = transformations[p++];
        Dtype ty = transformations[p++];
        Dtype g = transformations[p++];
        Dtype h = transformations[p++];
        Dtype i = transformations[p++];
        Dtype tz = transformations[p++];

        int z = t_grid_idx%grid_dim;
        int y = (t_grid_idx/grid_dim)%grid_dim;
        int x = t_grid_idx/yz;

        Dtype xx = x+0.5-c_offset;
        Dtype yy = y+0.5-c_offset;
        Dtype zz = z+0.5-c_offset;

        Dtype bx = a*xx + b*yy + c*zz + tx + c_offset;
        Dtype by = d*xx + e*yy + f*zz + ty + c_offset;
        Dtype bz = g*xx + h*yy + i*zz + tz + c_offset;

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
      } /* transformation_idx */
    } /* b_batch_idx */
  }
}

template <typename Dtype>
void Transform3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ForwardLabel(bottom[1], top[1]);
  if (output_inverse_transformations_) {
    ForwardInverseTransformations(&transformations_, top[2]);
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const vector<int>& bottom_shape = bottom[0]->shape();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int batch_size = bottom_shape[0];
  const int grid_dim = bottom_shape[1];
  const int num_grids = bottom[0]->count(1);
  const Dtype* transformations_data = transformations_.gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  Transform3DForward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(num_grids, grid_dim, batch_size, num_transformations_,
      pad_value_, bottom_data, transformations_data, top_data, len_transformation_param);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(Transform3DLayer);

}  // namespace caffe
