#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/normal_field_layer.hpp"

namespace caffe {

template<typename Dtype>
void NormalFieldLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalFieldLayer);

}  // namespace caffe
