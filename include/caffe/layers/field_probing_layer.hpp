#ifndef CAFFE_SPACE_TRAVEL_LAYER_HPP_
#define CAFFE_SPACE_TRAVEL_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief FieldProbingLayer
 */
template <typename Dtype>
class FieldProbingLayer : public Layer<Dtype> {
public:
  explicit FieldProbingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "FieldProbing"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int dim_grid_;
  int num_curve_;
  int len_curve_;
  bool output_normal_;
};

}  // namespace caffe

#endif  // CAFFE_SPACE_TRAVEL_LAYER_HPP_
