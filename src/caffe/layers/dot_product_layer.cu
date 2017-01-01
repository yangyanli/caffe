#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dot_product_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DotProductForward(const int num_filters, int num_sliding, const int batch_size, const int len_dot,
    const Dtype* weight, const Dtype* bias, const Dtype* bottom_data, Dtype* top_data, bool share_weights) {
  // One thread per filter per sliding position
  int num_sliding_filters = num_sliding*num_sliding*num_sliding*num_filters;
  CUDA_KERNEL_LOOP(sliding_filter_idx, num_sliding_filters) {
    int bias_offset = (share_weights?(sliding_filter_idx%num_filters):(sliding_filter_idx));
    int weight_offset = bias_offset*len_dot;

    int top_count = batch_size*num_sliding_filters;
    for (int top_offset = sliding_filter_idx; top_offset < top_count; top_offset += num_sliding_filters) {
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
void DotProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias = this->blobs_[1]->gpu_data();

  int num_sliding_total = num_sliding_*num_sliding_*num_sliding_;
  int num_sliding_filters = num_sliding_total * num_filter_;
  int len_dot = len_filter_ * num_channel_;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DotProductForward<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_filters), CAFFE_CUDA_NUM_THREADS>>>(
        num_filter_, num_sliding_, batch_size_, len_dot, weight, bias, bottom_data, top_data, share_weights_);
    CUDA_POST_KERNEL_CHECK;
  } /* bottom.size() */
}

template <typename Dtype>
__global__ void DotProductBackwardBias(const int num_sliding_filters, const int batch_size, const Dtype* top_diff, Dtype* bias_diff) {
  // One thread per filter per sliding position
  CUDA_KERNEL_LOOP(sliding_filter_idx, num_sliding_filters) {
    Dtype b_diff_sum = 0;
    int top_count = num_sliding_filters*batch_size;
    for (int top_offset = sliding_filter_idx; top_offset < top_count; top_offset += num_sliding_filters) {
      b_diff_sum += top_diff[top_offset];
    }
    bias_diff[sliding_filter_idx] += b_diff_sum;
  }
}

template <typename Dtype>
__global__ void DotProductBackwardWeightAndBottom(const int num_sliding_filters, const int batch_size, const int len_dot,
    const bool bp_weight, const bool bp_bottom,
    const Dtype* top_diff, const Dtype* weight, const Dtype* bottom_data, Dtype* weight_diff, Dtype* bottom_diff) {
  // One thread for each filter
  CUDA_KERNEL_LOOP(sliding_filter_idx, num_sliding_filters) {
    int top_count = num_sliding_filters*batch_size;
    for (int l = 0; l < len_dot; ++ l) {
      Dtype w_diff_sum = 0;
      int weight_offset = sliding_filter_idx*len_dot + l;
      for (int top_offset = sliding_filter_idx; top_offset < top_count; top_offset += num_sliding_filters) {
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
  // One thread per block element
  CUDA_KERNEL_LOOP(idx, block_size) {
    Dtype sum = 0;
    int total = block_size*block_count;
    for (int i = idx; i < total; i += block_size) {
      sum += before[i];
    }
    after[idx] = sum;
  }
}

template<typename Dtype>
void DotProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const vector<int>& bottom_shape = bottom[0]->shape();
  int batch_size = bottom_shape[0];
  int num_sliding_total = bottom_shape[1]*bottom_shape[2]*bottom_shape[3];
  int num_sliding_filters = num_sliding_total * num_filter_;
  int len_dot = len_filter_*num_channel_;
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
        DotProductBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_filters), CAFFE_CUDA_NUM_THREADS>>>(
            num_sliding_filters, batch_size, top_diff, slided_bias_diff);
        CUDA_POST_KERNEL_CHECK;
        // NOLINT_NEXT_LINE(whitespace/operators)
        BlockSum<Dtype><<<CAFFE_GET_BLOCKS(num_filter_), CAFFE_CUDA_NUM_THREADS>>>(num_filter_, num_sliding_total,
            slided_bias_diff, bias_diff);
        CUDA_POST_KERNEL_CHECK;
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        DotProductBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_filters), CAFFE_CUDA_NUM_THREADS>>>(
            num_sliding_filters, batch_size, top_diff, bias_diff);
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
        DotProductBackwardWeightAndBottom<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_filters), CAFFE_CUDA_NUM_THREADS>>>(
            num_sliding_filters, batch_size, len_dot, this->param_propagate_down_[0], propagate_down[i],
            top_diff, weight, bottom_data, slided_weight_diff, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
        int len_dot = len_filter_ * num_channel_;
        // NOLINT_NEXT_LINE(whitespace/operators)
        BlockSum<Dtype><<<CAFFE_GET_BLOCKS(num_filter_*len_dot), CAFFE_CUDA_NUM_THREADS>>>(num_filter_*len_dot,
            num_sliding_total, slided_weight_diff, weight_diff);
        CUDA_POST_KERNEL_CHECK;
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        DotProductBackwardWeightAndBottom<Dtype><<<CAFFE_GET_BLOCKS(num_sliding_filters), CAFFE_CUDA_NUM_THREADS>>>(
            num_sliding_filters, batch_size, len_dot, this->param_propagate_down_[0], propagate_down[i],
            top_diff, weight, bottom_data, weight_diff, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
      }
    }
  } /* top.size() */

  //Dtype scaler = 1.0/(batch_size*top.size());
  Dtype scaler = 1.0 / (top.size());
  caffe_gpu_scal(num_sliding_filters*len_dot, scaler, this->blobs_[0]->mutable_gpu_diff());
  caffe_gpu_scal(num_sliding_filters, scaler, this->blobs_[1]->mutable_gpu_diff());

  if (rand() % 100 == 0) {
    Dtype amax, aavg;

    caffe_gpu_amax(top[0]->count(), top[0]->gpu_diff(), &amax);
    caffe_gpu_aavg(top[0]->count(), top[0]->gpu_diff(), &aavg);
    LOG(INFO) << "DotProductLayer::Backward_gpu top_diff max-avg: " << amax << "\t" << aavg;

    caffe_gpu_amax(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), &amax);
    caffe_gpu_aavg(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), &aavg);
    LOG(INFO) << "DotProductLayer::Backward_gpu weight_diff max-avg: " << amax << "\t" << aavg;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DotProductLayer);

}  // namespace caffe
