#ifndef CAFFE_POOLING_4D_LAYER_HPP_
#define CAFFE_POOLING_4D_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class Pooling4DLayer : public Layer<Dtype> {
 public:
  explicit Pooling4DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling4D"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_s_, kernel_c_;
  int stride_s_, stride_c_;
  int pad_s_, pad_c_;
  int spatial_, channel_;
  int pooled_spatial_, pooled_channel_;
  Blob<unsigned int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_POOLING_4D_LAYER_HPP_
