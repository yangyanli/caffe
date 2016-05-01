#ifndef CAFFE_TRANSFORM_3D_LAYER_HPP_
#define CAFFE_TRANSFORM_3D_LAYER_HPP_

#include <boost/random.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Transform3DLayer : public Layer<Dtype> {
 public:
  explicit Transform3DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Transform3D"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinNumTopBlobs() const { return 2; }
  virtual inline int MaxNumTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  void ForwardLabel(Blob<Dtype>* input_labels, Blob<Dtype>* output_labels);

  typedef boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> > VariateGenerator;
  boost::shared_ptr<VariateGenerator> rotation_x_;
  boost::shared_ptr<VariateGenerator> rotation_y_;
  boost::shared_ptr<VariateGenerator> rotation_z_;
  boost::shared_ptr<VariateGenerator> translation_x_;
  boost::shared_ptr<VariateGenerator> translation_y_;
  boost::shared_ptr<VariateGenerator> translation_z_;
  boost::shared_ptr<VariateGenerator> scaling_x_;
  boost::shared_ptr<VariateGenerator> scaling_y_;
  boost::shared_ptr<VariateGenerator> scaling_z_;

  /*
  Layout of the transformation parameters
  a b c tx
  d e f ty
  g h i tz
  */
  static const int len_trans_params = 12;
  void GetTransformation(Dtype* transformation);
  void GetInverseTransformation(const Dtype* transformation, Dtype* inverse_transformation);
  void GetVariateGenerator(boost::shared_ptr<VariateGenerator>& vg, Dtype min, Dtype max);
  
  Dtype pad_value_;
  int num_transformations_;
  int batch_size_;
  std::string order_;
  Blob<Dtype> transformations_;
  Blob<Dtype> inverse_transformations_;
  bool output_inverse_transformations_;
};

}  // namespace caffe

#endif  // CAFFE_TRANSFORM_3D_LAYER_HPP_
