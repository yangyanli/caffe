#include "caffe/util/math_functions.hpp"
#include "caffe/layers/probing_curves_layer.hpp"

namespace caffe {

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const ProbingCurvesParameter& probing_curves_param = this->layer_param_.probing_curves_param();
  batch_size_ = (this->phase_ == TRAIN)?probing_curves_param.batch_size_train():probing_curves_param.batch_size_test();
  dim_grid_ = probing_curves_param.dim_grid();
  num_curve_ = probing_curves_param.num_curve();
  len_curve_ = probing_curves_param.len_curve();

  vector<int> weight_shape;
  weight_shape.push_back(dim_grid_);
  weight_shape.push_back(dim_grid_);
  weight_shape.push_back(dim_grid_);
  weight_shape.push_back(num_curve_);
  weight_shape.push_back(len_curve_);
  weight_shape.push_back(4);

  if (this->blobs_.size() > 0) {
    CHECK_EQ(1, this->blobs_.size()) << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape " << weight_shaped_blob.shape_string() << "; instead, shape was " << this->blobs_[0]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    Fill(this->blobs_[0].get(), probing_curves_param);

    Dtype* weight = this->blobs_[0]->mutable_cpu_data();
    int dim_field = probing_curves_param.dim_field();
    for (int i = 0; i < this->blobs_[0]->count()/4; i ++) {
      weight[i*4+0] *= dim_field;
      weight[i*4+1] *= dim_field;
      weight[i*4+2] *= dim_field;
      weight[i*4+3] = 1.0;
    }
  }
}

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const std::vector<int>& probing_curves_shape = this->blobs_[0]->shape();
  std::vector<int> top_shape;
  top_shape.push_back(batch_size_);
  top_shape.insert(top_shape.end(), probing_curves_shape.begin(), probing_curves_shape.end());

  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int weight_count = this->blobs_[0]->count();
  for (int i = 0; i < batch_size_; ++ i) {
    Dtype* top_data = top[0]->mutable_cpu_data()+weight_count*i;
    caffe_copy(weight_count, weight, top_data);
  }
}

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  int weight_count = this->blobs_[0]->count();
  caffe_set(weight_count, Dtype(0), weight_diff);
  for (int i = 0; i < batch_size_; ++ i) {
    const Dtype* top_diff = top[0]->cpu_diff()+weight_count*i;
    caffe_add(weight_count, top_diff, weight_diff, weight_diff);
  }
}

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::Fill(Blob<Dtype>* blob, const ProbingCurvesParameter& probing_curves_param) {
  int num_padding = probing_curves_param.num_padding();
  const vector<int>& shape = blob->shape();
  int dim_grid_x = shape[0];
  Dtype step_x = 1.0/(dim_grid_x+2*num_padding);
  int dim_grid_y = shape[1];
  Dtype step_y = 1.0/(dim_grid_y+2*num_padding);
  int dim_grid_z = shape[2];
  Dtype step_z = 1.0/(dim_grid_z+2*num_padding);
  int num_curve = shape[3];
  int len_curve = shape[4];
  Dtype min = probing_curves_param.min_span();
  Dtype max = probing_curves_param.max_span();
  if (min == max) {
    max = boost::math::float_next(min);
  }
  int max_ctl_num = probing_curves_param.max_ctl_num()+1;

  Dtype insphere_radius = std::min(std::min(step_x, step_y), step_z)/2;
  boost::uniform_real<Dtype> uniform_distribution_insphere_radius(0, insphere_radius);
  VariateGenerator rand_insphere_radius(caffe_rng(), uniform_distribution_insphere_radius);

  Dtype ctl_pt_radius = insphere_radius/2;
  boost::uniform_real<Dtype> uniform_distribution_ctl_pt_radius(0, ctl_pt_radius);
  VariateGenerator rand_ctl_pt_radius(caffe_rng(), uniform_distribution_ctl_pt_radius);

  boost::uniform_real<Dtype> uniform_distribution_sphere_surface(-1.0, 1.0);
  VariateGenerator rand_sphere_surface(caffe_rng(), uniform_distribution_sphere_surface);
  
  boost::uniform_real<Dtype> uniform_distribution_curve_length(min, max);
  VariateGenerator rand_curve_length(caffe_rng(), uniform_distribution_curve_length);

  std::vector<std::vector<int> > indices(dim_grid_x*dim_grid_y*dim_grid_z*num_curve, std::vector<int>(4, 0));
  int count = 0;
  for (int x = 0; x < dim_grid_x; ++ x) {
    for (int y = 0; y < dim_grid_y; ++ y) {
      for (int z = 0; z < dim_grid_z; ++ z) {
        for (int n = 0; n < num_curve; ++ n) {
          indices[count][0] = x;
          indices[count][1] = y;
          indices[count][2] = z;
          indices[count][3] = n;
          count ++;
        }
      }
    }
  }
  if (probing_curves_param.shuffle()) {
    std::random_shuffle(indices.begin(), indices.end());
  }
 
  count = 0;
  Dtype* data = blob->mutable_cpu_data();
  for (int x = 0; x < dim_grid_x; ++ x) {
    for (int y = 0; y < dim_grid_y; ++ y) {
      for (int z = 0; z < dim_grid_z; ++ z) {
        Dtype center_x = (x+num_padding+0.5)*step_x;
        Dtype center_y = (y+num_padding+0.5)*step_y;
        Dtype center_z = (z+num_padding+0.5)*step_z;
        for (int n = 0; n < num_curve; ++ n) {
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

          std::vector<int> index = indices[count++];
          index.push_back(0);
          index.push_back(0);
          int ctl_num = rand()%max_ctl_num;
          if (ctl_num == 0) {
            for (int l = 0; l < len_curve; ++ l) {
              Dtype ratio = 1.0*l/(len_curve-1);
              Dtype sample_x = start_x + (end_x-start_x)*ratio;
              Dtype sample_y = start_y + (end_y-start_y)*ratio;
              Dtype sample_z = start_z + (end_z-start_z)*ratio;

              index[4] = l;
              int offset = blob->offset(index);
              data[offset++] = sample_x;
              data[offset++] = sample_y;
              data[offset++] = sample_z;
            }
          } else {
            vector<Dtype> sample_points;
            GenerateCurve(sample_points, len_curve, ctl_num, start_x, start_y, start_z, end_x, end_y, end_z, rand_ctl_pt_radius, rand_sphere_surface);
            for (int l = 0; l < len_curve; ++ l) {
              Dtype xx = sample_points[l*3+0];
              Dtype yy = sample_points[l*3+1];
              Dtype zz = sample_points[l*3+2];
              ForceInRange(center_xx, center_yy, center_zz, xx, yy, zz);

              index[4] = l;
              int offset = blob->offset(index);
              data[offset++] = xx;
              data[offset++] = yy;
              data[offset++] = zz;
            }
          }
        }
      }
    }
  }
}

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::ForceInRange(const Dtype x, const Dtype y, const Dtype z, Dtype& xx, Dtype& yy, Dtype& zz) {
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
void ProbingCurvesLayer<Dtype>::SampleOnSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator) {
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
void ProbingCurvesLayer<Dtype>::SampleOnHalfSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator) {
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

template<typename Dtype>
void ProbingCurvesLayer<Dtype>::GenerateCurve(vector<Dtype> &sample_points, int sample_num, int ctl_num,
  Dtype sx, Dtype sy, Dtype sz, Dtype ex, Dtype ey, Dtype ez,
  VariateGenerator& vg_radius, VariateGenerator& vg_sphere_surface) {
  vector<Dtype> ctrl_points;

  Dtype dx = ex - sx; 
  Dtype dy = ey - sy; 
  Dtype dz = ez - sz; 

  Dtype len = sqrtf((dx * dx) + (dy * dy) + (dz * dz));

  dx /= len;
  dy /= len;
  dz /= len;

  ctrl_points.push_back(sx);
  ctrl_points.push_back(sy);
  ctrl_points.push_back(sz);
  Dtype step = len / (Dtype)(ctl_num+1);
  for(int i=1; i<=ctl_num; ++i) {
    Dtype cx = sx + i * step * dx;
    Dtype cy = sy + i * step * dy;
    Dtype cz = sz + i * step * dz;

    Dtype offset_x, offset_y, offset_z;
    SampleOnSphere(vg_radius(), offset_x, offset_y, offset_z, vg_sphere_surface);

    cx += offset_x;
    cy += offset_y;
    cz += offset_z;

    ctrl_points.push_back(cx);
    ctrl_points.push_back(cy);
    ctrl_points.push_back(cz);
  }
  ctrl_points.push_back(ex);
  ctrl_points.push_back(ey);
  ctrl_points.push_back(ez);

  Dtype delta = 1.0 / ((ctrl_points.size()/3)-1);
  Dtype sample_step = 1.0f / sample_num;

  for(Dtype t=0; t<=1.0f; t+=sample_step) {
    int p = (int)(t / delta);
    int s = (ctrl_points.size()/3) - 1;

    int p0 = p-1 < 0 ? 0 : p-1;
    int p1 = p;
    int p2 = p+1 > s ? s : p+1;
    int p3 = p+2 > s ? s : p+2;
    p0 *= 3;
    p1 *= 3;
    p2 *= 3;
    p3 *= 3;

    Dtype ax = ctrl_points[p0];
    Dtype ay = ctrl_points[p0+1];
    Dtype az = ctrl_points[p0+2];

    Dtype bx = ctrl_points[p1];
    Dtype by = ctrl_points[p1+1];
    Dtype bz = ctrl_points[p1+2];

    Dtype cx = ctrl_points[p2];
    Dtype cy = ctrl_points[p2+1];
    Dtype cz = ctrl_points[p2+2];

    Dtype dx = ctrl_points[p3];
    Dtype dy = ctrl_points[p3+1];
    Dtype dz = ctrl_points[p3+2];

    Dtype lt = (t - delta * p) / delta;

    Dtype t2 = lt * lt;
    Dtype t3 = t2 * lt;

    Dtype s1 = 0.5f * (  -t3 + 2*t2 - lt);
    Dtype s2 = 0.5f * ( 3*t3 - 5*t2 + 2);
    Dtype s3 = 0.5f * (-3*t3 + 4*t2 + lt);
    Dtype s4 = 0.5f * (   t3 -   t2    );

    ax *= s1;
    ay *= s1;
    az *= s1;

    bx *= s2;
    by *= s2;
    bz *= s2;

    cx *= s3;
    cy *= s3;
    cz *= s3;

    dx *= s4;
    dy *= s4;
    dz *= s4;

    Dtype rx = ax + bx + cx + dx;
    Dtype ry = ay + by + cy + dy;
    Dtype rz = az + bz + cz + dz;

    sample_points.push_back(rx);
    sample_points.push_back(ry);
    sample_points.push_back(rz);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ProbingCurvesLayer);
#endif

INSTANTIATE_CLASS(ProbingCurvesLayer);
REGISTER_LAYER_CLASS(ProbingCurves);

}  // namespace caffe
