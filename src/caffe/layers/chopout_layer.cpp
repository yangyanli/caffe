// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include <boost/random.hpp>

#include "caffe/util/rng.hpp"
#include "caffe/layers/chopout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// http://mathworld.wolfram.com/SpherePointPicking.html
template <typename Dtype>
void SampleOnSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >& variate_generator) {
  Dtype x1, x2, sqr_sum;
  do {
     x1 = variate_generator();
     x2 = variate_generator();
     sqr_sum = x1*x1 + x2*x2;
  } while (sqr_sum >= 1.0);
  x = 2*x1*std::sqrt(1-sqr_sum)*radius;
  y = 2*x2*std::sqrt(1-sqr_sum)*radius;
  z = (1-2*sqr_sum)*radius;
}

template <typename Dtype>
void ChopoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4) << "ChopoutLayer supports only 3D data.";
  const vector<int>& bottom_shape = bottom[0]->shape();
  bool is_cube = (bottom_shape[1] == bottom_shape[2] && bottom_shape[1] == bottom_shape[3]);
  CHECK_EQ(is_cube, true) << "ChopoutLayer supports only cube shape data.";

  const ChopoutParameter& chopout_param = this->layer_param_.chopout_param();
  min_chop_radius_ = chopout_param.min_chop_radius();
  max_chop_radius_ = chopout_param.max_chop_radius();
  num_chop_mask_ = chopout_param.num_chop_mask();
  pad_value_ = chopout_param.pad_value();
  vector<int> chop_masks_shape = bottom_shape;
  chop_masks_shape[0] = num_chop_mask_;
  chop_masks_.Reshape(chop_masks_shape);

  vector<int> mask_indices_shape;
  mask_indices_shape.push_back(bottom_shape[0]);
  mask_indices_.Reshape(mask_indices_shape);

  typedef boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> > VariateGenerator;
  boost::uniform_real<Dtype> uniform_distribution_sphere_surface(-1.0, 1.0);
  VariateGenerator rand_sphere_surface(caffe_rng(), uniform_distribution_sphere_surface);

  boost::uniform_real<Dtype> uniform_distribution_radius(min_chop_radius_, max_chop_radius_);
  VariateGenerator rand_center_radius(caffe_rng(), uniform_distribution_radius);

  int grid_dim = bottom_shape[1];
  Dtype scaler = 1.0/(grid_dim-1);
  unsigned int* chop_masks = chop_masks_.mutable_cpu_data();
  for (int i = 0; i < num_chop_mask_; ++ i) {
    Dtype radius = rand_center_radius();
    Dtype a, b, c;
    SampleOnSphere(radius, a, b, c, rand_sphere_surface);
    Dtype x = a + 0.5;
    Dtype y = b + 0.5;
    Dtype z = c + 0.5;
    Dtype d = -(a*x + b*y + c*z);

    Dtype keep_ratio = 0;
    Dtype c_side = a*0.5 + b*0.5 + c*0.5 + d;
    for (int gx = 0; gx < grid_dim; ++ gx) {
      Dtype xx = gx*scaler;
      for (int gy = 0; gy < grid_dim; ++ gy) {
        Dtype yy = gy*scaler;
        for (int gz = 0; gz < grid_dim; ++ gz) {
          Dtype zz = gz*scaler;
          Dtype p_side = a*xx + b*yy + c*zz + d;
          bool keep = (c_side*p_side >= 0);
          chop_masks[chop_masks_.offset(i, gx, gy, gz)] = keep;
          keep_ratio += keep;
        }
      }
    }
  }
}

template <typename Dtype>
void ChopoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void ChopoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int batch_size = bottom[0]->shape(0);
  const int num_grids = bottom[0]->count(1);
  if (this->phase_ == TRAIN) {
    const unsigned int* chop_masks = chop_masks_.cpu_data();
    {
      unsigned int* mask_indices = mask_indices_.mutable_cpu_data();
      for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        mask_indices[batch_idx] = rand()%num_chop_mask_;
      }
    }

    const unsigned int* mask_indices = mask_indices_.cpu_data();
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      int n_offset = batch_idx * num_grids;
      int m_idx = mask_indices[batch_idx];
      int m_offset = m_idx * num_grids;
      for (int grid_idx = 0; grid_idx < num_grids; ++grid_idx) {
        int top_offset = n_offset + grid_idx;
        if (chop_masks[m_offset+grid_idx]) { 
          top_data[top_offset] = bottom_data[top_offset];
        } else {
          top_data[top_offset] = pad_value_;
        }
      } /* num_grids */
    } /* batch_size */
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ChopoutLayer);
#endif

INSTANTIATE_CLASS(ChopoutLayer);
REGISTER_LAYER_CLASS(Chopout);

}  // namespace caffe
