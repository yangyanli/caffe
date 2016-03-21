#include <algorithm>
#include <vector>

#include "caffe/layers/gaussian_layer.hpp"

namespace caffe {

template<typename Dtype>
void GaussianLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  x_and_gaussian_x_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GaussianLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* x_data = x_and_gaussian_x_.mutable_cpu_data();
  Dtype* gaussian_x_data = x_and_gaussian_x_.mutable_cpu_diff();
  const int count = bottom[0]->count();
  Dtype sigma = this->layer_param_.gaussian_param().sigma();
  Dtype multiplier = -1.0/(2.0*sigma*sigma);
  for (int i = 0; i < count; ++i) {
    const Dtype& x = bottom_data[i];
    Dtype gaussian_x = std::exp(x*x*multiplier);
    top_data[i] = gaussian_x;
    x_data[i] = x;
    gaussian_x_data[i] = gaussian_x;
  }
}

template <typename Dtype>
void GaussianLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* x_data = x_and_gaussian_x_.cpu_data();
    const Dtype* gaussian_x_data = x_and_gaussian_x_.cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype sigma = this->layer_param_.gaussian_param().sigma();
    Dtype multiplier = -1.0/(sigma*sigma);
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * x_data[i] * gaussian_x_data[i] * multiplier;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GaussianLayer);
#endif

INSTANTIATE_CLASS(GaussianLayer);
REGISTER_LAYER_CLASS(Gaussian);


}  // namespace caffe
