#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/curv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CurvolutionForward(const int num_all_curves, const int batch_size, const int len_dot,
    const Dtype* weight, const Dtype* bias, const Dtype* bottom_data, Dtype* top_data) {
  int curve_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each curve
  if(curve_idx < num_all_curves) {
    int top_count = num_all_curves*batch_size;
    for (int top_offset = curve_idx; top_offset < top_count; top_offset += num_all_curves) {
      Dtype dot = 0;
      int bottom_offset = top_offset*len_dot;
      int weight_offset = curve_idx*len_dot;
      for (int l = 0; l < len_dot; ++ l) {
        dot += bottom_data[bottom_offset++]*weight[weight_offset++];
      }
      top_data[top_offset] = dot + bias[curve_idx];
    }
  }
}

template<typename Dtype>
void CurvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias = this->blobs_[1]->gpu_data();

  const vector<int>& bottom_shape = bottom[0]->shape();
  int batch_size = bottom_shape[0];
  int num_grids = bottom_shape[1]*bottom_shape[2]*bottom_shape[3];
  int num_all_curves = num_grids * num_curve_;
  int len_dot = len_curve_*num_channel_;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    CurvolutionForward<Dtype><<<CAFFE_GET_BLOCKS(num_all_curves), CAFFE_CUDA_NUM_THREADS>>>(
    num_all_curves, batch_size, len_dot, weight, bias, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
  } /* bottom.size() */
}

template <typename Dtype>
__global__ void CurvolutionBackwardBias(const int num_all_curves, const int batch_size, const Dtype* top_diff, Dtype* bias_diff) {
  int curve_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each curve
  if(curve_idx < num_all_curves) {
    Dtype b_diff_sum = 0;
    int top_count = num_all_curves*batch_size;
    for (int top_offset = curve_idx; top_offset < top_count; top_offset += num_all_curves) {
      b_diff_sum += top_diff[top_offset];
    }
    bias_diff[curve_idx] += b_diff_sum;
  }
}

template <typename Dtype>
__global__ void CurvolutionBackwardWeightAndBottom(const int num_all_curves, const int batch_size, const int len_dot,
    const bool bp_weight, const bool bp_bottom,
    const Dtype* top_diff, const Dtype* weight, const Dtype* bottom_data, Dtype* weight_diff, Dtype* bottom_diff) {
  int curve_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each curve
  if(curve_idx < num_all_curves) {
    int top_count = num_all_curves*batch_size;
    for (int l = 0; l < len_dot; ++ l) {
      Dtype w_diff_sum = 0;
      int weight_offset = curve_idx*len_dot + l;
      for (int top_offset = curve_idx; top_offset < top_count; top_offset += num_all_curves) {
        const Dtype& t_diff = top_diff[top_offset];
        int bottom_offset = top_offset*len_dot + l;
        w_diff_sum += t_diff*bottom_data[bottom_offset];
        if(bp_bottom) {
          bottom_diff[bottom_offset] += t_diff*weight[weight_offset];
        }
      }
      if(bp_weight) {
        weight_diff[weight_offset] += w_diff_sum;
      }
    }
  }
}

template<typename Dtype>
void CurvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const vector<int>& bottom_shape = bottom[0]->shape();
  int batch_size = bottom_shape[0];
  int num_grids = bottom_shape[1]*bottom_shape[2]*bottom_shape[3];
  int num_all_curves = num_grids * num_curve_;
  int len_dot = len_curve_*num_channel_;
  if (this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      // NOLINT_NEXT_LINE(whitespace/operators)
      CurvolutionBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(num_all_curves), CAFFE_CUDA_NUM_THREADS>>>(
          num_all_curves, batch_size, top_diff, bias_diff);
      CUDA_POST_KERNEL_CHECK;
    }
  }

  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom_diff);
      // NOLINT_NEXT_LINE(whitespace/operators)
      CurvolutionBackwardWeightAndBottom<Dtype><<<CAFFE_GET_BLOCKS(num_all_curves), CAFFE_CUDA_NUM_THREADS>>>(
          num_all_curves, batch_size, len_dot, this->param_propagate_down_[0], propagate_down[i],
          top_diff, weight, bottom_data, weight_diff, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    }
    if(rand()%100 == 0){
      Dtype amax, aavg;
      caffe_gpu_amax(top[i]->count(), top_diff, &amax);
      caffe_gpu_aavg(top[i]->count(), top_diff, &aavg);
      LOG(INFO) << "CurvolutionLayer::Backward_gpu (" << num_channel_ << ") top_diff max-avg: " << amax << "\t" << aavg;
    }
  } /* top.size() */
  Dtype scaler = 1.0/(batch_size*top.size());
  caffe_gpu_scal(num_all_curves*len_dot, scaler, this->blobs_[0]->mutable_gpu_diff());
  caffe_gpu_scal(num_all_curves, scaler, this->blobs_[1]->mutable_gpu_diff());

  if(rand()%100 == 0){
    Dtype amax, aavg;
    caffe_gpu_amax(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), &amax);
    caffe_gpu_aavg(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), &aavg);
    LOG(INFO) << "CurvolutionLayer::Backward_gpu (" << num_channel_ << ") weight_diff max-avg: " << amax << "\t" << aavg;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CurvolutionLayer);

}  // namespace caffe
