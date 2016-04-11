#ifndef CAFFE_NORMAL_FIELD_LAYER_HPP_
#define CAFFE_NORMAL_FIELD_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief NormalFieldLayer
 */
template <typename Dtype>
class NormalFieldLayer : public Layer<Dtype> {
public:
  explicit NormalFieldLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "NormalField"; }
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

#endif  // CAFFE_NORMAL_FIELD_LAYER_HPP_
