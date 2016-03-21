#include <vector>

#include "caffe/layers/chopout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ChopoutForward(const int num_grids, const int batch_size, const Dtype pad_value, const Dtype* bottom_data,
    const unsigned int* chop_masks, const unsigned int* mask_indices, Dtype* top_data) {
  int grid_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each grid
  if(grid_idx < num_grids) {
    int top_count = num_grids*batch_size;
    for (int top_offset = grid_idx, batch_idx = 0; top_offset < top_count; top_offset += num_grids) {
      int m_idx = mask_indices[batch_idx];
      if (chop_masks[m_idx*num_grids+grid_idx]) {
        top_data[top_offset] = bottom_data[top_offset];
      } else {
        top_data[top_offset] = pad_value;
      }

      batch_idx ++;
    }
  }
}

template <typename Dtype>
void ChopoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int batch_size = bottom[0]->shape(0);
  const int num_grids = bottom[0]->count(1);
  if (this->phase_ == TRAIN) {
    const unsigned int* chop_masks = chop_masks_.gpu_data();
    {
      unsigned int* mask_indices = mask_indices_.mutable_cpu_data();
      for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        mask_indices[batch_idx] = rand()%num_chop_mask_;
      }
    }

    const unsigned int* mask_indices = mask_indices_.gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ChopoutForward<Dtype><<<CAFFE_GET_BLOCKS(num_grids), CAFFE_CUDA_NUM_THREADS>>>(
        num_grids, batch_size, pad_value_, bottom_data, chop_masks, mask_indices, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ChopoutLayer);

}  // namespace caffe
