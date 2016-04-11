#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/normal_field_layer.hpp"

namespace caffe {

template<typename Dtype>
void NormalFieldLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4) << "NormalFieldLayer supports only 3D data.";
  const vector<int>& bottom_shape = bottom[0]->shape();
  bool is_cube = (bottom_shape[1] == bottom_shape[2] && bottom_shape[1] == bottom_shape[3]);
  CHECK_EQ(is_cube, true) << "NormalFieldLayer supports only cube shape data.";
}

template<typename Dtype>
void NormalFieldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape.push_back(3);
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void NormalFieldLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

#ifdef CPU_ONLY
STUB_GPU(NormalFieldLayer);
#endif

INSTANTIATE_CLASS(NormalFieldLayer);
REGISTER_LAYER_CLASS(NormalField);

}  // namespace caffe
