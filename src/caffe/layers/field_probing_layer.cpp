#include <boost/random.hpp>
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/field_probing_layer.hpp"

namespace caffe {

template<typename Dtype>
const int FieldProbingLayer<Dtype>::len_coordinates;
template<typename Dtype>
const int FieldProbingLayer<Dtype>::len_trans_params;

template<typename Dtype>
void FieldProbingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  field_num_ = top.size();
  transform_ = (bottom.size() != field_num_);
  if (transform_) {
    CHECK_EQ(bottom.size(), field_num_ + 1) << "FieldProbingLayer takes only one optional transformation bottom.";
  } else {
    CHECK_EQ(bottom.size(), field_num_) << "FieldProbingLayer expects equal number of input fields and outputs.";
  }

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

#ifndef CPU_ONLY
  std::vector<int> slided_trans_shape;
  slided_trans_shape.push_back(num_curve_);
  slided_trans_shape.push_back(len_curve_);
  slided_trans_shape.push_back(batch_size_);
  slided_trans_shape.push_back(len_trans_params);
  slided_trans_.Reshape(slided_trans_shape);

#endif // !CPU_ONLY
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape;
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
  int field_dim_1 = field_dim_ - 1;
  int num_samples = num_curve_ * len_curve_;
  int slided_num_samples = num_sliding_ * num_sliding_ * num_sliding_ * num_samples;
  Dtype step = field_dim_ * 1.0 / num_sliding_;

  const Dtype* filters = this->blobs_[0]->cpu_data();
  const Dtype* trans = transform_ ? (bottom[field_num_]->cpu_data()) : (NULL);
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
          if (!transform_) {
            x = sx; y = sy; z = sz;
            SnapGrid_cpu(x, x0, x1, x_a, x_m, field_dim_1);
            SnapGrid_cpu(y, y0, y1, y_a, y_m, field_dim_1);
            SnapGrid_cpu(z, z0, z1, z_a, z_m, field_dim_1);
          }

          for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
            if (transform_) {
              const Dtype* t = trans + batch_idx * len_trans_params;
              x = t[0] * sx + t[1] * sy + t[2] * sz + t[3];
              y = t[4] * sx + t[5] * sy + t[6] * sz + t[7];
              z = t[8] * sx + t[9] * sy + t[10] * sz + t[11];
              SnapGrid_cpu(x, x0, x1, x_a, x_m, field_dim_1);
              SnapGrid_cpu(y, y0, y1, y_a, y_m, field_dim_1);
              SnapGrid_cpu(z, z0, z1, z_a, z_m, field_dim_1);
            }

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
  int field_dim_1 = field_dim_ - 1;
  int num_samples = num_curve_ * len_curve_;
  int num_sliding_total = num_sliding_ * num_sliding_ * num_sliding_;
  int slided_num_samples = num_sliding_total * num_samples;
  Dtype step = field_dim_ * 1.0 / num_sliding_;

  const Dtype* filters = this->blobs_[0]->cpu_data();
  const Dtype* trans = transform_ ? (bottom[field_num_]->cpu_data()) : (NULL);

  Dtype* filters_diff = this->blobs_[0]->mutable_cpu_diff();
  caffe_set(this->blobs_[0]->count(), Dtype(0), filters_diff);
  Dtype* trans_diff = transform_ ? (bottom[field_num_]->mutable_cpu_diff()) : (NULL);
  if(transform_) {
    caffe_set(bottom[field_num_]->count(), Dtype(0), trans_diff);
  }

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
          if (!transform_) {
            x = sx; y = sy; z = sz;
            SnapGrid_cpu(x, x0, x1, x_a, x_m, field_dim_1);
            SnapGrid_cpu(y, y0, y1, y_a, y_m, field_dim_1);
            SnapGrid_cpu(z, z0, z1, z_a, z_m, field_dim_1);
          }

          Dtype w_diff_x, w_diff_y, w_diff_z;
          w_diff_x = w_diff_y = w_diff_z = 0;

          for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
            if (transform_) {
              const Dtype* t = trans + batch_idx * len_trans_params;
              x = t[0] * sx + t[1] * sy + t[2] * sz + t[3];
              y = t[4] * sx + t[5] * sy + t[6] * sz + t[7];
              z = t[8] * sx + t[9] * sy + t[10] * sz + t[11];
              SnapGrid_cpu(x, x0, x1, x_a, x_m, field_dim_1);
              SnapGrid_cpu(y, y0, y1, y_a, y_m, field_dim_1);
              SnapGrid_cpu(z, z0, z1, z_a, z_m, field_dim_1);
            }

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

            if (transform_) {
              const Dtype* t = trans + batch_idx * len_trans_params;
              Dtype t_diff_x = t[0]*diff_x + t[4]*diff_y + t[8]*diff_z;
              Dtype t_diff_y = t[1]*diff_x + t[5]*diff_y + t[9]*diff_z;
              Dtype t_diff_z = t[2]*diff_x + t[6]*diff_y + t[10]*diff_z;

              Dtype* t_diff = trans_diff + batch_idx * len_trans_params;
              t_diff[0] += x*diff_x;
              t_diff[1] += y*diff_x;
              t_diff[2] += z*diff_x;
              t_diff[3] += diff_x;
              t_diff[4] += x*diff_y;
              t_diff[5] += y*diff_y;
              t_diff[6] += z*diff_y;
              t_diff[7] += diff_y;
              t_diff[8] += x*diff_z;
              t_diff[9] += y*diff_z;
              t_diff[10] += z*diff_z;
              t_diff[11] += diff_z;

              w_diff_x += t_diff_x;
              w_diff_x += t_diff_y;
              w_diff_x += t_diff_z;
            } else {
              w_diff_x += diff_x;
              w_diff_x += diff_y;
              w_diff_x += diff_z;
            }
          } /* batch_idx */

          filters_diff[p_offset + 0] += w_diff_x;
          filters_diff[p_offset + 1] += w_diff_y;
          filters_diff[p_offset + 2] += w_diff_z;

          sliding_idx++;
        } /* k */
      } /* j */
    } /* i */
  } /* sample_idx */


  caffe_scal(this->blobs_[0]->count(), Dtype(1.0/(num_sliding_total*batch_size_*field_num_)), filters_diff);
  if (transform_) {
    caffe_scal(bottom[field_num_]->count(), Dtype(1.0/(slided_num_samples*field_num_)), trans_diff);
  }

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
    if (transform_) {
      Dtype amax = caffe_cpu_amax(bottom[field_num_]->count(), bottom[field_num_]->cpu_diff());
      Dtype aavg = caffe_cpu_aavg(bottom[field_num_]->count(), bottom[field_num_]->cpu_diff());
      LOG(INFO) << "FieldProbingLayer::Backward_cpu trans_diff max-avg: " << amax << "\t" << aavg;
    }
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::InitializeFilters(Blob<Dtype>* blob, const FieldProbingParameter& param) {
  Dtype radius = field_dim_ * 0.5 / num_sliding_ + padding_;
  Dtype min_span = 2 * param.min_span() * radius;
  Dtype max_span = 2 * param.max_span() * radius;

  boost::uniform_real<Dtype> uniform_distribution_cube_point(-0.5 * radius, 0.5 * radius);
  typedef boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> > VariateGenerator;
  VariateGenerator rand_cube_point(caffe_rng(), uniform_distribution_cube_point);

  Dtype* data = blob->mutable_cpu_data();
  int count = 0;
  while (count < num_curve_) {
    Dtype sx = rand_cube_point();
    Dtype sy = rand_cube_point();
    Dtype sz = rand_cube_point();
    while (true) {
      Dtype ex = rand_cube_point();
      Dtype ey = rand_cube_point();
      Dtype ez = rand_cube_point();
      Dtype dx = ex - sx;
      Dtype dy = ey - sy;
      Dtype dz = ez - sz;
      Dtype length = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (length <= max_span && length >= min_span) {
        for (int l = 0; l < len_curve_; ++l) {
          Dtype ratio = 1.0 * l / (len_curve_ - 1);
          Dtype sample_x = sx + dx * ratio;
          Dtype sample_y = sy + dy * ratio;
          Dtype sample_z = sz + dz * ratio;

          data[count * len_coordinates + 0] = sample_x;
          data[count * len_coordinates + 1] = sample_y;
          data[count * len_coordinates + 2] = sample_z;
          data[count * len_coordinates + 3] = 1.0;
        }
        count++;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FieldProbingLayer);
#endif

INSTANTIATE_CLASS(FieldProbingLayer);
REGISTER_LAYER_CLASS(FieldProbing);

}  // namespace caffe
