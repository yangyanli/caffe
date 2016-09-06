#include <boost/random.hpp>
#include <boost/math/special_functions/next.hpp>

#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/field_probing_layer.hpp"

namespace caffe {

template<typename Dtype>
const int FieldProbingLayer<Dtype>::len_coordinates;

template<typename Dtype>
void FieldProbingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  field_num_ = top.size();

  std::vector<int> field_0_shape = bottom[0]->shape();
  CHECK(field_0_shape.size() == 4 || field_0_shape.size() == 5) << "FieldProbingLayer supports only 4D or 5D data.";
  batch_size_ = field_0_shape[0];
  field_dim_ = field_0_shape[1];
  CHECK(field_dim_ == field_0_shape[2] && field_dim_ == field_0_shape[3]) << "FieldProbingLayer supports only cube fields.";

  if (field_0_shape.size() == 5) {
    field_0_shape.pop_back();
  }
  for (int i = 0; i < field_num_; ++i) {
    std::vector<int> field_i_shape = bottom[i]->shape();
    if (field_i_shape.size() == 5) {
      field_i_shape.pop_back();
    }
    CHECK(field_0_shape == field_i_shape) << "All input fields must be in the same shape.";
  }

  const FieldProbingParameter& param = this->layer_param_.field_probing_param();
  num_sliding_ = param.num_sliding();
  padding_ = param.padding();
  num_curve_ = param.num_curve();
  len_curve_ = param.len_curve();

  std::vector<int> filters_shape;
  filters_shape.push_back(num_curve_);
  filters_shape.push_back(len_curve_);
  filters_shape.push_back(len_coordinates);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1, this->blobs_.size()) << "Incorrect number of weight blobs.";
    if (filters_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> filters_shaped_blob(filters_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape " << filters_shaped_blob.shape_string()
          << "; instead, shape was " << this->blobs_[0]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(filters_shape));
    InitializeFilters(this->blobs_[0].get(), param);
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape;
  top_shape.push_back(batch_size_);
  top_shape.push_back(num_sliding_);
  top_shape.push_back(num_sliding_);
  top_shape.push_back(num_sliding_);
  top_shape.push_back(num_curve_);
  top_shape.push_back(len_curve_);

  top_shape.push_back(0);
  for (int i = 0; i < field_num_; ++i) {
    const std::vector<int>& field_shape = bottom[i]->shape();
    int field_channels = (field_shape.size() == 5) ? (field_shape.back()) : (1);
    top_shape.back() = field_channels;
    top[i]->Reshape(top_shape);
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num_samples = num_curve_ * len_curve_;
  int slided_num_samples = num_sliding_ * num_sliding_ * num_sliding_ * num_samples;
  Dtype step = field_dim_ * 1.0 / num_sliding_;
  int field_dim_1 = field_dim_ - 1;

  const Dtype* filters = this->blobs_[0]->cpu_data();
  for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    int p_offset = sample_idx * len_coordinates;
    Dtype px = filters[p_offset + 0];
    Dtype py = filters[p_offset + 1];
    Dtype pz = filters[p_offset + 2];

    int sliding_idx = 0;
    for (int i = 0; i < num_sliding_; ++i) {
      Dtype sx = px + (i + 0.5) * step;
      for (int j = 0; j < num_sliding_; ++j) {
        Dtype sy = py + (j + 0.5) * step;
        for (int k = 0; k < num_sliding_; ++k) {
          Dtype sz = pz + (k + 0.5) * step;

          Dtype x, y, z;
          int x0, y0, z0, x1, y1, z1;
          Dtype x_a, y_a, z_a, x_m, y_m, z_m;
          x = sx; y = sy; z = sz;
          SnapGrid_cpu(x, x0, x1, x_a, x_m, field_dim_1);
          SnapGrid_cpu(y, y0, y1, y_a, y_m, field_dim_1);
          SnapGrid_cpu(z, z0, z1, z_a, z_m, field_dim_1);

          for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
            int top_offset = batch_idx * slided_num_samples + sliding_idx * num_samples + sample_idx;
            for (int field_idx = 0; field_idx < field_num_; ++field_idx) {
              const std::vector<int>& field_shape = bottom[i]->shape();
              int field_channels = (field_shape.size() == 5) ? (field_shape.back()) : (1);

              const Dtype* bottom_data = bottom[field_idx]->cpu_data();
              Dtype* t_data = top[field_idx]->mutable_cpu_data() + top_offset * field_channels;
              Interpolate_cpu(bottom_data, batch_idx, x, y, z, x0, y0, z0, x1, y1, z1, field_dim_, t_data, field_channels);
            }
          } /* batch_idx */
          sliding_idx++;
        } /* k */
      } /* j */
    } /* i */

  } /* sample_idx */
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num_samples = num_curve_ * len_curve_;
  int num_sliding_total = num_sliding_ * num_sliding_ * num_sliding_;
  int slided_num_samples = num_sliding_total * num_samples;
  Dtype step = field_dim_ * 1.0 / num_sliding_;
  int field_dim_1 = field_dim_ - 1;

  const Dtype* filters = this->blobs_[0]->cpu_data();
  Dtype* filters_diff = this->blobs_[0]->mutable_cpu_diff();
  caffe_set(this->blobs_[0]->count(), Dtype(0), filters_diff);

  for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    int p_offset = sample_idx * len_coordinates;
    Dtype px = filters[p_offset + 0];
    Dtype py = filters[p_offset + 1];
    Dtype pz = filters[p_offset + 2];

    int sliding_idx = 0;
    for (int i = 0; i < num_sliding_; ++i) {
      Dtype sx = px + (i + 0.5) * step;
      for (int j = 0; j < num_sliding_; ++j) {
        Dtype sy = py + (j + 0.5) * step;
        for (int k = 0; k < num_sliding_; ++k) {
          Dtype sz = pz + (k + 0.5) * step;

          Dtype x, y, z;
          int x0, y0, z0, x1, y1, z1;
          Dtype x_a, y_a, z_a, x_m, y_m, z_m;
          x = sx; y = sy; z = sz;
          SnapGrid_cpu(x, x0, x1, x_a, x_m, field_dim_1);
          SnapGrid_cpu(y, y0, y1, y_a, y_m, field_dim_1);
          SnapGrid_cpu(z, z0, z1, z_a, z_m, field_dim_1);

          Dtype w_diff_x, w_diff_y, w_diff_z;
          w_diff_x = w_diff_y = w_diff_z = 0;

          for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
            Dtype diff_x, diff_y, diff_z;
            diff_x = diff_y = diff_z = 0;
            int top_offset = batch_idx * slided_num_samples + sliding_idx * num_samples + sample_idx;
            for (int field_idx = 0; field_idx < field_num_; ++field_idx) {
              const Dtype* bottom_data = bottom[field_idx]->cpu_data();
              const std::vector<int>& field_shape = bottom[field_idx]->shape();
              int field_channels = (field_shape.size() == 5) ? (field_shape.back()) : (1);

              const Dtype* top_diff = top[field_idx]->cpu_diff() + top_offset * field_channels;
              Dtype* gradients = new Dtype[field_channels * 3];
              ComputeGradient_cpu(bottom_data, batch_idx, x, y, z, x0, y0, z0, x1, y1, z1,
                  x_a, y_a, z_a, x_m, y_m, z_m, field_dim_, gradients, field_channels);
              for (int channel_idx = 0; channel_idx < field_channels; ++channel_idx) {
                diff_x += top_diff[channel_idx] * gradients[3 * channel_idx + 0];
                diff_y += top_diff[channel_idx] * gradients[3 * channel_idx + 1];
                diff_z += top_diff[channel_idx] * gradients[3 * channel_idx + 2];
              }
              delete gradients;
            }
            w_diff_x += diff_x;
            w_diff_y += diff_y;
            w_diff_z += diff_z;
          } /* batch_idx */

          filters_diff[p_offset + 0] += w_diff_x;
          filters_diff[p_offset + 1] += w_diff_y;
          filters_diff[p_offset + 2] += w_diff_z;

          sliding_idx++;
        } /* k */
      } /* j */
    } /* i */
  } /* sample_idx */


  //caffe_scal(this->blobs_[0]->count(), Dtype(1.0/(num_sliding_total*batch_size_*field_num_)), filters_diff);
  caffe_scal(this->blobs_[0]->count(), Dtype(1.0/(num_sliding_total*field_num_)), filters_diff);
  if (rand() % 100 == 0) {
    {
      Dtype amax = caffe_cpu_amax(top[0]->count(), top[0]->cpu_diff());
      Dtype aavg = caffe_cpu_aavg(top[0]->count(), top[0]->cpu_diff());
      LOG(INFO) << "FieldProbingLayer::Backward_cpu top_diff max-avg: " << amax << "\t" << aavg;
    }
    {
      Dtype amax = caffe_cpu_amax(this->blobs_[0]->count(), this->blobs_[0]->cpu_diff());
      Dtype aavg = caffe_cpu_aavg(this->blobs_[0]->count(), this->blobs_[0]->cpu_diff());
      LOG(INFO) << "FieldProbingLayer::Backward_cpu weight_diff max-avg: " << amax << "\t" << aavg;
    }
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::InitializeFilters(Blob<Dtype>* blob, const FieldProbingParameter& param) {
  const int dim_grid = 8;
  const int num_grid = dim_grid*dim_grid*dim_grid;

  Dtype radius = field_dim_ * 0.5 / num_sliding_ + padding_;
  Dtype diameter = 2 * radius;

  int num_padding = 1;
  Dtype step = 1.0/(dim_grid+2*num_padding);
  Dtype min = param.min_span();
  Dtype max = param.max_span();
  if (min == max) {
    max = boost::math::float_next(min);
  }

  Dtype insphere_radius = step/2;
  boost::uniform_real<Dtype> uniform_distribution_insphere_radius(0, insphere_radius);
  VariateGenerator rand_insphere_radius(caffe_rng(), uniform_distribution_insphere_radius);

  Dtype ctl_pt_radius = insphere_radius/2;
  boost::uniform_real<Dtype> uniform_distribution_ctl_pt_radius(0, ctl_pt_radius);
  VariateGenerator rand_ctl_pt_radius(caffe_rng(), uniform_distribution_ctl_pt_radius);

  boost::uniform_real<Dtype> uniform_distribution_sphere_surface(-1.0, 1.0);
  VariateGenerator rand_sphere_surface(caffe_rng(), uniform_distribution_sphere_surface);
  
  boost::uniform_real<Dtype> uniform_distribution_curve_length(min, max);
  VariateGenerator rand_curve_length(caffe_rng(), uniform_distribution_curve_length);

  int curve_count = 0;
  Dtype* data = blob->mutable_cpu_data();
  while(curve_count < num_curve_) {
    int idx = rand()%num_grid;
    int x = idx%dim_grid;
    int y = (idx/dim_grid)%dim_grid;
    int z = idx/(dim_grid*dim_grid);
    Dtype center_x = (x+num_padding+0.5)*step;
    Dtype center_y = (y+num_padding+0.5)*step;
    Dtype center_z = (z+num_padding+0.5)*step;

    Dtype std_radius = rand_insphere_radius();
    Dtype std_x, std_y, std_z;
    SampleOnSphere(std_radius, std_x, std_y, std_z, rand_sphere_surface);

    Dtype offset_radius = rand_curve_length();
    Dtype offset_x, offset_y, offset_z;
    SampleOnSphere(offset_radius, offset_x, offset_y, offset_z, rand_sphere_surface);

    Dtype center_xx = center_x + std_x;
    Dtype center_yy = center_y + std_y;
    Dtype center_zz = center_z + std_z;

    Dtype start_x = center_xx + offset_x;
    Dtype start_y = center_yy + offset_y;
    Dtype start_z = center_zz + offset_z;
    ForceInRange(center_xx, center_yy, center_zz, start_x, start_y, start_z);

    Dtype end_x = center_xx - offset_x;
    Dtype end_y = center_yy - offset_y;
    Dtype end_z = center_zz - offset_z;
    ForceInRange(center_xx, center_yy, center_zz, end_x, end_y, end_z);

    for (int l = 0; l < len_curve_; ++ l) {
      Dtype ratio = 1.0*l/(len_curve_-1);
      Dtype sample_x = start_x + (end_x-start_x)*ratio;
      Dtype sample_y = start_y + (end_y-start_y)*ratio;
      Dtype sample_z = start_z + (end_z-start_z)*ratio;

      std::vector<int> index(3, 0);
      index[0] = curve_count; index[1] = l;
      int offset = blob->offset(index);
      data[offset++] = sample_x*diameter-radius;
      data[offset++] = sample_y*diameter-radius;
      data[offset++] = sample_z*diameter-radius;
    } 
    curve_count ++;
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::ForceInRange(const Dtype x, const Dtype y, const Dtype z, Dtype& xx, Dtype& yy, Dtype& zz) {
  if(xx < 0.0) {
    Dtype offset_x = x-xx;
    Dtype offset_y = y-yy;
    Dtype offset_z = z-zz;
    Dtype ratio = -xx/offset_x;
    xx = 0.0;
    yy += ratio*offset_y;
    zz += ratio*offset_z;
  }
  if(yy < 0.0) {
    Dtype offset_x = x-xx;
    Dtype offset_y = y-yy;
    Dtype offset_z = z-zz;
    Dtype ratio = -yy/offset_y;
    xx += ratio*offset_x;
    yy = 0.0;
    zz += ratio*offset_z;
  }
  if(zz < 0.0) {
    Dtype offset_x = x-xx;
    Dtype offset_y = y-yy;
    Dtype offset_z = z-zz;
    Dtype ratio = -zz/offset_z;
    xx += ratio*offset_x;
    yy += ratio*offset_y;
    zz = 0.0;
  }

  if(xx > 1.0) {
    Dtype offset_x = x-xx;
    Dtype offset_y = y-yy;
    Dtype offset_z = z-zz;
    Dtype ratio = (1.0-xx)/offset_x;
    xx = 1.0;
    yy += ratio*offset_y;
    zz += ratio*offset_z;
  }
  if(yy > 1.0) {
    Dtype offset_x = x-xx;
    Dtype offset_y = y-yy;
    Dtype offset_z = z-zz;
    Dtype ratio = (1.0-yy)/offset_y;
    xx += ratio*offset_x;
    yy = 1.0;
    zz += ratio*offset_z;
  }
  if(zz > 1.0) {
    Dtype offset_x = x-xx;
    Dtype offset_y = y-yy;
    Dtype offset_z = z-zz;
    Dtype ratio = (1.0-zz)/offset_z;
    xx += ratio*offset_x;
    yy += ratio*offset_y;
    zz = 1.0;
  }
}

// http://mathworld.wolfram.com/SpherePointPicking.html
template<typename Dtype>
void FieldProbingLayer<Dtype>::SampleOnSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator) {
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

template<typename Dtype>
void FieldProbingLayer<Dtype>::SampleOnHalfSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator) {
  Dtype x1, x2, sqr_sum;
  do {
     x1 = variate_generator();
     x2 = variate_generator();
     sqr_sum = x1*x1 + x2*x2;
  } while (sqr_sum >= 1.0 || x1 > 0);
  x = 2*x1*std::sqrt(1-sqr_sum)*radius;
  y = 2*x2*std::sqrt(1-sqr_sum)*radius;
  z = (1-2*sqr_sum)*radius;
}

#ifdef CPU_ONLY
STUB_GPU(FieldProbingLayer);
#endif

INSTANTIATE_CLASS(FieldProbingLayer);
REGISTER_LAYER_CLASS(FieldProbing);

}  // namespace caffe
