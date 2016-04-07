#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/probing_curves_layer.hpp"

namespace caffe {

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int weight_count = this->blobs_[0]->count();
  for (int i = 0; i < batch_size_; ++ i) {
    Dtype* top_data = top[0]->mutable_gpu_data()+weight_count*i;
    caffe_copy(weight_count, weight, top_data);
  }
}

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  int weight_count = this->blobs_[0]->count();
  caffe_gpu_set(weight_count, Dtype(0), weight_diff);
  for (int i = 0; i < batch_size_; ++ i) {
    const Dtype* top_diff = top[0]->gpu_diff()+weight_count*i;
    caffe_gpu_add(weight_count, top_diff, weight_diff, weight_diff);
  }
  caffe_gpu_scal(weight_count, Dtype(1.0/batch_size_), weight_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(ProbingCurvesLayer);

}  // namespace caffe
