#ifndef CAFFE_TRANSFORM_3D_LAYER_HPP_
#define CAFFE_TRANSFORM_3D_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

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
  virtual inline int ExactNumTopBlobs() const { return 2; }

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

  void PrepareMappingAndRotations(void);
  void ForwardLabel(Blob<Dtype>* input_labels, Blob<Dtype>* output_labels);

  Dtype min_rotation_x_;
  Dtype min_rotation_y_;
  Dtype min_rotation_z_;
  Dtype max_rotation_x_;
  Dtype max_rotation_y_;
  Dtype max_rotation_z_;

  Dtype min_translation_x_;
  Dtype min_translation_y_;
  Dtype min_translation_z_;
  Dtype max_translation_x_;
  Dtype max_translation_y_;
  Dtype max_translation_z_;

  Dtype min_scaling_x_;
  Dtype min_scaling_y_;
  Dtype min_scaling_z_;
  Dtype max_scaling_x_;
  Dtype max_scaling_y_;
  Dtype max_scaling_z_;

  Dtype pad_value_;

  int num_transformations_;
  Blob<Dtype> transformations_;
};

}  // namespace caffe

#endif  // CAFFE_TRANSFORM_3D_LAYER_HPP_
