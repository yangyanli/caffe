#ifndef CAFFE_GRADIENT_FIELD_LAYER_HPP_
#define CAFFE_GRADIENT_FIELD_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief GradientFieldLayer
 */
template <typename Dtype>
class GradientFieldLayer : public Layer<Dtype> {
public:
  explicit GradientFieldLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "GradientField"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactBottomBlobs() const { return 1; }
  virtual inline int ExactTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
};

}  // namespace caffe

#endif  // CAFFE_GRADIENT_FIELD_LAYER_HPP_
