#include "caffe/layers/transformer_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
void Transformer3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batch_size = bottom[0]->shape()[0]; 
  int probing_curves_size = bottom[0]->count(1);
  int num_samples = probing_curves_size/4;
  CHECK_EQ(batch_size, bottom[1]->shape()[0])
      << "Transformations and probing curves must have the same batch size.";
  CHECK_EQ(bottom[1]->count(1), len_transformation_param)
      << "Transformation must have " << len_transformation_param << " parameters.";
  std::vector<int> temp_diff_shape = bottom[1]->shape();
  temp_diff_shape.push_back(num_samples);
  temp_diff_.Reshape(temp_diff_shape);
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
      top_data[0] = (t[0]+1)*x + t[1]*y + t[2]*z + t[3]; 
      top_data[1] = t[4]*x + (t[5]+1)*y + t[6]*z + t[7]; 
      top_data[2] = t[8]*x + t[9]*y + (t[10]+1)*z + t[11]; 
    } 
  }
}

template <typename Dtype>
void Transformer3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
  int batch_size = bottom[0]->shape()[0];
  int probing_curves_size = bottom[0]->count(1);
  int num_samples = probing_curves_size/4;
  for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
    const Dtype* t = bottom[1]->cpu_data()+len_transformation_param*batch_idx;
    Dtype* t_diff = bottom[1]->mutable_cpu_diff()+len_transformation_param*batch_idx;
    for (int sample_idx = 0; sample_idx < num_samples; ++ sample_idx) {
      int offset = (batch_idx*num_samples + sample_idx)*4;
      const Dtype* bottom_data = bottom[0]->cpu_data()+offset;
      const Dtype& x = bottom_data[0];
      const Dtype& y = bottom_data[1];
      const Dtype& z = bottom_data[2];

      const Dtype* top_diff = top[0]->cpu_diff()+offset;
      const Dtype& x_d = top_diff[0];
      const Dtype& y_d = top_diff[1];
      const Dtype& z_d = top_diff[2];

      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff()+offset;
      bottom_diff[0] = (t[0]+1)*x_d + t[4]*y_d + t[8]*z_d;
      bottom_diff[1] = t[1]*x_d + (t[5]+1)*y_d + t[9]*z_d;
      bottom_diff[2] = t[2]*x_d + t[6]*y_d + (t[10]+1)*z_d;
      
      t_diff[0] += x*x_d;
      t_diff[1] += y*x_d;
      t_diff[2] += z*x_d;
      t_diff[3] += x_d;
      t_diff[4] += x*y_d;
      t_diff[5] += y*y_d;
      t_diff[6] += z*y_d;
      t_diff[7] += y_d;
      t_diff[8] += x*z_d;
      t_diff[9] += y*z_d;
      t_diff[10] += z*z_d;
      t_diff[11] += z_d;
    }
  }
  caffe_scal(batch_size*len_transformation_param, Dtype(1.0/num_samples), bottom[1]->mutable_cpu_diff());
}


#ifdef CPU_ONLY
STUB_GPU(Transformer3DLayer);
#endif

INSTANTIATE_CLASS(Transformer3DLayer);
REGISTER_LAYER_CLASS(Transformer3D);

}  // namespace caffe
