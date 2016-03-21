// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_CURVES_FILLER_HPP
#define CAFFE_CURVES_FILLER_HPP

#include <string>
#include <boost/random.hpp>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtpe>
class Filler;

template <typename Dtype>
class CurvesFiller : public Filler<Dtype> {
 public:
  typedef boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> > VariateGenerator;

  explicit CurvesFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
     
 virtual void Fill(Blob<Dtype>* blob) {
    const vector<int>& shape = blob->shape();
    int dim_grid_x = shape[0];
    Dtype step_x = 1.0/(dim_grid_x+2);
    int dim_grid_y = shape[1];
    Dtype step_y = 1.0/(dim_grid_y+2);
    int dim_grid_z = shape[2];
    Dtype step_z = 1.0/(dim_grid_z+2);
    int num_curve = shape[3];
    int len_curve = shape[4];
    Dtype min = this->filler_param_.min();
    Dtype max = this->filler_param_.max();
    int max_ctl_pt_num = ((int)(abs(this->filler_param_.value()))+1);

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
 
    Dtype* data = blob->mutable_cpu_data();
    vector<int> index;
    for (int x = 0; x < dim_grid_x; ++ x) {
      index.push_back(x);
      for (int y = 0; y < dim_grid_y; ++ y) {
        index.push_back(y);
        for (int z = 0; z < dim_grid_z; ++ z) {
          index.push_back(z);
          Dtype center_x = (x+1.5)*step_x;
          Dtype center_y = (y+1.5)*step_y;
          Dtype center_z = (z+1.5)*step_z;
          for (int n = 0; n < num_curve; ++ n) {
            index.push_back(n);

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

            int ctl_pt_num = rand()%max_ctl_pt_num;
            if (ctl_pt_num == 0) {
              for (int l = 0; l < len_curve; ++ l) {
                index.push_back(l);

                Dtype ratio = 1.0*l/(len_curve-1);
                Dtype sample_x = start_x + (end_x-start_x)*ratio;
                Dtype sample_y = start_y + (end_y-start_y)*ratio;
                Dtype sample_z = start_z + (end_z-start_z)*ratio;

                index.push_back(0);
                data[blob->offset(index)] = sample_x;
                index.pop_back();

                index.push_back(1);
                data[blob->offset(index)] = sample_y;
                index.pop_back();

                index.push_back(2);
                data[blob->offset(index)] = sample_z;
                index.pop_back();

                index.pop_back();
              }
            } else {
              vector<Dtype> sample_points;
              GenerateCurve(sample_points, len_curve, ctl_pt_num, start_x, start_y, start_z, end_x, end_y, end_z, rand_ctl_pt_radius, rand_sphere_surface);
              for (int l = 0; l < len_curve; ++ l) {
                index.push_back(l);

                Dtype xx = sample_points[l*3+0];
                Dtype yy = sample_points[l*3+1];
                Dtype zz = sample_points[l*3+2];
                ForceInRange(center_xx, center_yy, center_zz, xx, yy, zz);

                index.push_back(0);
                data[blob->offset(index)] = xx;
                index.pop_back();

                index.push_back(1);
                data[blob->offset(index)] = yy;
                index.pop_back();

                index.push_back(2);
                data[blob->offset(index)] = zz;
                index.pop_back();

                index.pop_back();

              }
            }
            index.pop_back();
          }
          index.pop_back();
        }
        index.pop_back();
      }
      index.pop_back();
    }
  }

 private:
  void ForceInRange(const Dtype x, const Dtype y, const Dtype z, Dtype& xx, Dtype& yy, Dtype& zz) {
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
  void SampleOnSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator) {
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
  void SampleOnHalfSphere(Dtype radius, Dtype& x, Dtype& y, Dtype& z, VariateGenerator& variate_generator) {
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
 
  void GenerateCurve(vector<Dtype> &sample_points, int sample_num, int ctl_pt_num,
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
    Dtype step = len / (Dtype)(ctl_pt_num+1);
    for(int i=1; i<=ctl_pt_num; ++i) {
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
};

}  // namespace caffe

#endif  // CAFFE_CURVES_FILLER_HPP_
