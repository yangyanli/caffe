#include "caffe/layers/transformer_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
void Transformer3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(Transformer3DLayer);

}  // namespace caffe
