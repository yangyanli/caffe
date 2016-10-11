#include "caffe/util/field_operations.hpp"

#include "caffe/layers/transform_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Transform3DForward(const int num_grids, const int field_dim, const int batch_size, const int num_transformations,
    const Dtype pad_value, const Dtype* bottom_data, const Dtype* transformations, Dtype* top_data,
    const int len_trans_params, const int field_channels) {
  const int t_grid_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each grid
  if(t_grid_idx < num_grids) {
    int field_dim_1 = field_dim-1;
    Dtype c_offset = field_dim_1/2.0;
    const int yz = field_dim*field_dim;
    for (int b_batch_idx = 0; b_batch_idx < batch_size; ++ b_batch_idx) {
      int offset = b_batch_idx * num_transformations;
      for(int transformation_idx = 0; transformation_idx < num_transformations; ++ transformation_idx) {
        int t_batch_idx = offset + transformation_idx;

        int z = t_grid_idx%field_dim;
        int y = (t_grid_idx/field_dim)%field_dim;
        int x = t_grid_idx/yz;

        Dtype xx = x-c_offset;
        Dtype yy = y-c_offset;
        Dtype zz = z-c_offset;

        const Dtype* t = transformations+t_batch_idx*len_trans_params;
        Dtype bx = t[0]*xx + t[1]*yy + t[2]*zz + t[3] + c_offset;
        Dtype by = t[4]*xx + t[5]*yy + t[6]*zz + t[7] + c_offset;
        Dtype bz = t[8]*xx + t[9]*yy + t[10]*zz + t[11] + c_offset;

        Dtype* t_data = top_data + (t_batch_idx*num_grids+t_grid_idx)*field_channels;
        if(bx >= 0 && bx < field_dim
            && by >= 0 && by < field_dim
            && bz >= 0 && bz < field_dim) {
          int x0, y0, z0, x1, y1, z1;
          SnapGrid_gpu(bx, x0, x1, field_dim_1);
          SnapGrid_gpu(by, y0, y1, field_dim_1);
          SnapGrid_gpu(bz, z0, z1, field_dim_1);
          Interpolate_gpu(bottom_data, b_batch_idx, bx, by, bz, x0, y0, z0, x1, y1, z1,
            field_dim,t_data, field_channels);
        } else {
          for (int i = 0; i < field_channels; ++ i) {
            t_data[i] = pad_value;
          }
        }
      } /* transformation_idx */
    } /* b_batch_idx */
  }
}

template <typename Dtype>
void Transform3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int field_num = bottom.size()-1;
  ForwardLabel(bottom[field_num], top[field_num]);

  int num_output = batch_size_*num_transformations_;
  if (output_transformations_) {
    caffe_copy(num_output*len_trans_params, transformations_.gpu_data(), top[field_num+1]->mutable_gpu_data());
  }

  for (int i = 0; i < field_num; ++ i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const vector<int>& field_shape = bottom[i]->shape();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int field_dim = field_shape[1];
    const int num_grids = field_dim*field_dim*field_dim;
    int field_channels = (field_shape.size() == 5)?(field_shape.back()):(1);
    const Dtype* transformations_data = transformations_.gpu_data();
  
    // NOLINT_NEXT_LINE(whitespace/operators)
    Transform3DForward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(num_grids, field_dim, batch_size_, num_transformations_,
        pad_values_[i], bottom_data, transformations_data, top_data, len_trans_params, field_channels);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Transform3DLayer);

}  // namespace caffe
