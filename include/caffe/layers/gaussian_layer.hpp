#ifndef CAFFE_GAUSSIAN_LAYER_HPP_
#define CAFFE_GAUSSIAN_LAYER_HPP_

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template<typename Dtype>
class GaussianLayer: public NeuronLayer<Dtype> {
public:
  explicit GaussianLayer(const LayerParameter& param)
  :NeuronLayer<Dtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "Gaussian";
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> x_and_gaussian_x_;
};

}  // namespace caffe

#endif  // CAFFE_GAUSSIAN_LAYER_HPP_
