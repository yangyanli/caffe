#include "caffe/layers/transformer_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
void Transformer3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(Transformer3DLayer);
#endif

INSTANTIATE_CLASS(Transformer3DLayer);
REGISTER_LAYER_CLASS(Transformer3D);

}  // namespace caffe
