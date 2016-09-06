#ifndef CAFFE_DOT_PRODUCT_LAYER_HPP_
#define CAFFE_DOT_PRODUCT_LAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief DotProductLayer
 */
template<typename Dtype>
class DotProductLayer: public Layer<Dtype> {
public:
  explicit DotProductLayer(const LayerParameter& param)
  :
      Layer<Dtype>(param) {
  }
  virtual inline const char* type() const {
    return "DotProduct";
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const {
    return 1;
  }
  virtual inline int MinTopBlobs() const {
    return 1;
  }
  virtual inline bool EqualNumBottomTopBlobs() const {
    return true;
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

  int batch_size_;
  int num_sliding_;
  int num_filter_;
  int len_filter_;
  int num_channel_;
  bool share_weights_;

#ifndef CPU_ONLY
  Blob<Dtype> slided_weight_;
  Blob<Dtype> slided_bias_;
#endif // !CPU_ONLY
};

}  // namespace caffe

#endif  // CAFFE_DOT_PRODUCT_LAYER_HPP_
