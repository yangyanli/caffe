#ifndef CAFFE_PROBING_CURVES_LAYER_HPP_
#define CAFFE_PROBING_CURVES_LAYER_HPP_

#include <boost/random.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief ProbingCurvesLayer
 */
template <typename Dtype>
class ProbingCurvesLayer : public Layer<Dtype> {
public:
  explicit ProbingCurvesLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "ProbingCurves"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactBottomBlobs() const { return 0; }
  virtual inline int ExactTopBlobs() const { return 1; }

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
  int dim_grid_;
  int num_curve_;
  int len_curve_;

private:
  typedef boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> > VariateGenerator;
  static void Fill(Blob<Dtype>* blob, const ProbingCurvesParameter& probing_curve_param);
  static void ForceInRange(const Dtype x, const Dtype y, const Dtype z, Dtype& xx, Dtype& yy, Dtype& zz);
  static void SampleOnSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator);
  static void SampleOnHalfSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator);
  static void GenerateCurve(vector<Dtype> &sample_points, int sample_num, int ctl_pt_num,
    Dtype sx, Dtype sy, Dtype sz, Dtype ex, Dtype ey, Dtype ez,
    VariateGenerator& vg_radius, VariateGenerator& vg_sphere_surface);

};

}  // namespace caffe

#endif  // CAFFE_PROBING_CURVES_LAYER_HPP_
