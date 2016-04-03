#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/probing_curves_layer.hpp"

namespace caffe {

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_copy(this->blobs_[0]->count(), weight, top_data);
}

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_copy(this->blobs_[0]->count(), top_diff, weight_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(ProbingCurvesLayer);

}  // namespace caffe
