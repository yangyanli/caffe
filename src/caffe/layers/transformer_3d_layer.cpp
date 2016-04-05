#include "caffe/layers/transformer_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
void Transformer3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape()[0], bottom[1]->shape()[0])
      << "Transformations and probing curves must have the same batch size.";
  CHECK_EQ(bottom[1]->count(1), len_transformation_param)
      << "Transformation must have " << len_transformation_param << " parameters.";
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batch_size = bottom[0]->shape()[0]; 
  int probing_curves_size = bottom[0]->count(1);
  int num_samples = probing_curves_size/4;
  for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
    const Dtype* t = bottom[1]->cpu_data()+len_transformation_param*batch_idx;
    for (int sample_idx = 0; sample_idx < num_samples; ++ sample_idx) {
      int offset = (batch_idx*num_samples + sample_idx)*4;
      const Dtype* bottom_data = bottom[0]->cpu_data()+offset;
      const Dtype& x = bottom_data[0];
      const Dtype& y = bottom_data[1];
      const Dtype& z = bottom_data[2];

      Dtype* top_data = top[0]->mutable_cpu_data()+offset;
      top_data[0] = t[0]*x + t[1]*y + t[2]*z + t[3]; 
      top_data[1] = t[4]*x + t[5]*y + t[6]*z + t[7]; 
      top_data[2] = t[8]*x + t[9]*y + t[10]*z + t[11]; 
    } 
  }
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int batch_size = bottom[0]->shape()[0];
  int probing_curves_size = bottom[0]->count(1);
  int num_samples = probing_curves_size/4;
  for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
    const Dtype* t = bottom[1]->cpu_data()+len_transformation_param*batch_idx;
    for (int sample_idx = 0; sample_idx < num_samples; ++ sample_idx) {
      int offset = (batch_idx*num_samples + sample_idx)*4;
      const Dtype* bottom_data = bottom[0]->cpu_data()+offset;
      const Dtype& x = bottom_data[0];
      const Dtype& y = bottom_data[1];
      const Dtype& z = bottom_data[2];

    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(Transformer3DLayer);
#endif

INSTANTIATE_CLASS(Transformer3DLayer);
REGISTER_LAYER_CLASS(Transformer3D);

}  // namespace caffe
