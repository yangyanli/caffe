#ifndef CAFFE_FIELD_PROBING_LAYER_HPP_
#define CAFFE_FIELD_PROBING_LAYER_HPP_

#include <boost/random.hpp>
#include "caffe/util/rng.hpp"
#include "caffe/layer.hpp"

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
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void InitializeFilters(Blob<Dtype>* blob, const FieldProbingParameter& param);
  typedef boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> > VariateGenerator;
  static void ForceInRange(const Dtype x, const Dtype y, const Dtype z, Dtype& xx, Dtype& yy, Dtype& zz);
  static void SampleOnSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator);
  static void SampleOnHalfSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator);

  int field_num_;
  int batch_size_;
  int field_dim_;

  int num_sliding_;
  Dtype padding_;

  int num_curve_;
  int len_curve_;

  static const int len_coordinates = 4;
};

}  // namespace caffe

#endif  // CAFFE_FIELD_PROBING_LAYER_HPP_
