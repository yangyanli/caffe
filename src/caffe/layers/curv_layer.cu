#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/curv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CurvolutionForward(const int num_curves, int num_sliding, const int batch_size, const int len_dot,
    const Dtype* weight, const Dtype* bias, const Dtype* bottom_data, Dtype* top_data, bool share_weights) {
  int sliding_curve_idx = blockDim.x*blockIdx.x + threadIdx.x;
  int num_sliding_curves = num_sliding*num_sliding*num_sliding*num_curves;
  // One thread per curve per sliding position
  if(sliding_curve_idx < num_sliding_curves) {
    int bias_offset = (share_weights?(sliding_curve_idx%num_curves):(sliding_curve_idx));
    int weight_offset = bias_offset*len_dot;

    int top_count = batch_size*num_sliding_curves;
    for (int top_offset = sliding_curve_idx; top_offset < top_count; top_offset += num_sliding_curves) {
      Dtype dot = 0;
      int bottom_offset = top_offset*len_dot;
      for (int l = 0; l < len_dot; ++ l) {
        dot += bottom_data[bottom_offset+l]*weight[weight_offset+l];
      }
      top_data[top_offset] = dot + bias[bias_offset];
    }
  }
}

template<typename Dtype>
void CurvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias = this->blobs_[1]->gpu_data();

  int num_sliding_total = num_sliding_*num_sliding_*num_sliding_;
  int num_sliding_curves = num_sliding_total * num_curve_;
  int len_dot = len_curve_ * num_channel_;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    CurvolutionForward<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_curves), CAFFE_CUDA_NUM_THREADS>>>(
        num_curve_, num_sliding_, batch_size_, len_dot, weight, bias, bottom_data, top_data, share_weights_);
    CUDA_POST_KERNEL_CHECK;
  } /* bottom.size() */
}

template <typename Dtype>
__global__ void CurvolutionBackwardBias(const int num_sliding_curves, const int batch_size, const Dtype* top_diff, Dtype* bias_diff) {
  int sliding_curve_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread per curve per sliding position
  if(sliding_curve_idx < num_sliding_curves) {
    Dtype b_diff_sum = 0;
    int top_count = num_sliding_curves*batch_size;
    for (int top_offset = sliding_curve_idx; top_offset < top_count; top_offset += num_sliding_curves) {
      b_diff_sum += top_diff[top_offset];
    }
    bias_diff[sliding_curve_idx] += b_diff_sum;
  }
}

template <typename Dtype>
__global__ void CurvolutionBackwardWeightAndBottom(const int num_sliding_curves, const int batch_size, const int len_dot,
    const bool bp_weight, const bool bp_bottom,
    const Dtype* top_diff, const Dtype* weight, const Dtype* bottom_data, Dtype* weight_diff, Dtype* bottom_diff) {
  int sliding_curve_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread for each curve
  if(sliding_curve_idx < num_sliding_curves) {
    int top_count = num_sliding_curves*batch_size;
    for (int l = 0; l < len_dot; ++ l) {
      Dtype w_diff_sum = 0;
      int weight_offset = sliding_curve_idx*len_dot + l;
      for (int top_offset = sliding_curve_idx; top_offset < top_count; top_offset += num_sliding_curves) {
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

template <typename Dtype>
__global__ void BlockSum(const int block_size, const int block_count, const Dtype* before, Dtype* after) {
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  // One thread per block element
  if(idx < block_size) {
    Dtype sum = 0;
    int total = block_size*block_count;
    for (int i = idx; i < total; i += block_size) {
      sum += before[i];
    }
    after[idx] = sum;
  }
}

template<typename Dtype>
void CurvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const vector<int>& bottom_shape = bottom[0]->shape();
  int batch_size = bottom_shape[0];
  int num_sliding_total = bottom_shape[1]*bottom_shape[2]*bottom_shape[3];
  int num_sliding_curves = num_sliding_total * num_curve_;
  int len_dot = len_curve_*num_channel_;
  Dtype* bias_diff = NULL;
  if (this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->param_propagate_down_[1]) {
      if(share_weights_) {
        Dtype* slided_bias_diff = slided_bias_.mutable_gpu_diff();
        caffe_gpu_set(slided_bias_.count(), Dtype(0), slided_bias_diff);
        // NOLINT_NEXT_LINE(whitespace/operators)
        CurvolutionBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_curves), CAFFE_CUDA_NUM_THREADS>>>(
            num_sliding_curves, batch_size, top_diff, slided_bias_diff);
        CUDA_POST_KERNEL_CHECK;
        // NOLINT_NEXT_LINE(whitespace/operators)
        BlockSum<Dtype><<<CAFFE_GET_BLOCKS(num_curve_), CAFFE_CUDA_NUM_THREADS>>>(num_curve_, num_sliding_total,
            slided_bias_diff, bias_diff);
        CUDA_POST_KERNEL_CHECK;
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        CurvolutionBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_curves), CAFFE_CUDA_NUM_THREADS>>>(
            num_sliding_curves, batch_size, top_diff, bias_diff);
        CUDA_POST_KERNEL_CHECK;
      }
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
      if(share_weights_) {
        Dtype* slided_weight_diff = slided_weight_.mutable_gpu_diff();
        caffe_gpu_set(slided_weight_.count(), Dtype(0), slided_weight_diff);
        // NOLINT_NEXT_LINE(whitespace/operators)
        CurvolutionBackwardWeightAndBottom<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_curves), CAFFE_CUDA_NUM_THREADS>>>(
            num_sliding_curves, batch_size, len_dot, this->param_propagate_down_[0], propagate_down[i],
            top_diff, weight, bottom_data, slided_weight_diff, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
        int len_dot = len_curve_ * num_channel_;
        // NOLINT_NEXT_LINE(whitespace/operators)
        BlockSum<Dtype><<<CAFFE_GET_BLOCKS(num_curve_*len_dot), CAFFE_CUDA_NUM_THREADS>>>(num_curve_*len_dot,
            num_sliding_total, slided_weight_diff, weight_diff);
        CUDA_POST_KERNEL_CHECK;
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        CurvolutionBackwardWeightAndBottom<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_curves), CAFFE_CUDA_NUM_THREADS>>>(
            num_sliding_curves, batch_size, len_dot, this->param_propagate_down_[0], propagate_down[i],
            top_diff, weight, bottom_data, weight_diff, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
      }
    }
  } /* top.size() */

  //Dtype scaler = 1.0/(batch_size*top.size());
  Dtype scaler = 1.0 / (top.size());
  caffe_gpu_scal(num_sliding_curves*len_dot, scaler, this->blobs_[0]->mutable_gpu_diff());
  caffe_gpu_scal(num_sliding_curves, scaler, this->blobs_[1]->mutable_gpu_diff());

  if (rand() % 100 == 0) {
    Dtype amax, aavg;

    caffe_gpu_amax(top[0]->count(), top[0]->gpu_diff(), &amax);
    caffe_gpu_aavg(top[0]->count(), top[0]->gpu_diff(), &aavg);
    LOG(INFO) << "CurvolutionLayer::Backward_gpu top_diff max-avg: " << amax << "\t" << aavg;

    caffe_gpu_amax(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), &amax);
    caffe_gpu_aavg(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), &aavg);
    LOG(INFO) << "CurvolutionLayer::Backward_gpu weight_diff max-avg: " << amax << "\t" << aavg;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CurvolutionLayer);

}  // namespace caffe
