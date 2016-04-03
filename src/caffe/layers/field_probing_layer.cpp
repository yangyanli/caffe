#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/field_probing_layer.hpp"

namespace caffe {

template<typename Dtype>
void FieldProbingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int bottom_id = 2; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[1]->shape() == bottom[bottom_id]->shape())
        << "All input fields must have the same shape.";
  }

  const std::vector<int>& probing_curves_shape = bottom[0]->shape();
  dim_grid_ = probing_curves_shape[0];
  //dim_grid_ = probing_curves_shape[1];
  //dim_grid_ = probing_curves_shape[2];
  num_curve_ = probing_curves_shape[3];
  len_curve_ = probing_curves_shape[4];

  output_normal_ = (top.size() == 2*(bottom.size()-1));
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int batch_size = bottom[1]->shape(0);
  const vector<int>& probing_curves_shape = bottom[0]->shape();

  vector<int> top_shape;
  top_shape.push_back(batch_size);
  top_shape.insert(top_shape.end(), probing_curves_shape.begin(), probing_curves_shape.end());
  top_shape.back() = 1;

  vector<int> top_normal_shape = top_shape;
  top_normal_shape.back() = 3;
  for (int i = 1; i < bottom.size(); ++ i) {
    if(output_normal_) {
      top[2*(i-1)+0]->Reshape(top_shape);
      top[2*(i-1)+1]->Reshape(top_normal_shape);
    } else {
      top[i-1]->Reshape(top_shape);
    }
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* probing_curves = bottom[0]->cpu_data();

  const vector<int>& field_shape = bottom[1]->shape();
  int batch_size = field_shape[0];
  int field_dim_x = field_shape[1];
  int field_dim_y = field_shape[2];
  int field_dim_z = field_shape[3];
  int field_dim_x_1 = field_dim_x-1;
  int field_dim_y_1 = field_dim_y-1;
  int field_dim_z_1 = field_dim_z-1;
  int num_grids = dim_grid_*dim_grid_*dim_grid_;
  int num_samples = num_grids*num_curve_*len_curve_;

  for (int i = 1; i < bottom.size(); ++ i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = NULL;
    Dtype* top_normal_data = NULL;
    if(output_normal_) {
      top_data = top[2*i+0]->mutable_cpu_data();
      top_normal_data = top[2*i+1]->mutable_cpu_data();
    } else {
      top_data = top[i]->mutable_cpu_data();
    }

    for (int grid_idx = 0; grid_idx < num_grids; ++ grid_idx) {
      int g_offset = grid_idx * num_curve_ * len_curve_;
      for (int c = 0; c < num_curve_; ++ c) {
        int c_offset = c * len_curve_;
        for (int l = 0; l < len_curve_; ++ l) {
          int sample_idx = (g_offset + c_offset + l);
          int p_offset = sample_idx * 4;
          Dtype x = probing_curves[p_offset+0];
          Dtype y = probing_curves[p_offset+1];
          Dtype z = probing_curves[p_offset+2];
          int x0, y0, z0, x1, y1, z1;
          Dtype x_a, y_a, z_a, x_m, y_m, z_m;
          SnapGrid_cpu(x, x0, x1, x_a, x_m, field_dim_x_1);
          SnapGrid_cpu(y, y0, y1, y_a, y_m, field_dim_y_1);
          SnapGrid_cpu(z, z0, z1, z_a, z_m, field_dim_z_1);
          Dtype x_x0 = x-x0;
          Dtype y_y0 = y-y0;
          Dtype z_z0 = z-z0;
          Dtype x1_x = x1-x;
          Dtype y1_y = y1-y;
          Dtype z1_z = z1-z;
          for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
            int n_offset = batch_idx * num_samples;
            int top_offset = n_offset + g_offset + c_offset + l;
            if(output_normal_) {
              Dtype nx, ny, nz;
              top_data[top_offset] = ComputeGradient_cpu(bottom_data, batch_idx,
                  x0, y0, z0, x1, y1, z1,
                  x_a, y_a, z_a, x_m, y_m, z_m,
                  x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
                  nx, ny, nz, field_dim_x, field_dim_y, field_dim_z);
              Normalize_cpu(nx, ny, nz);
              top_normal_data[top_offset*3+0] = nx;
              top_normal_data[top_offset*3+1] = ny;
              top_normal_data[top_offset*3+2] = nz;
            } else {
              top_data[top_offset] = Interpolate_cpu(bottom_data, batch_idx,
                  x0, y0, z0, x1, y1, z1,
                  x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
                  field_dim_x, field_dim_y, field_dim_z);
            }
          } /* batch_size */
        } /* len_curve */
      } /* num_curve */
    } /* num_grids */
  } /* bottom.size() */
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* probing_curves = bottom[0]->cpu_data();
  Dtype* probing_curves_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), probing_curves_diff);

  const vector<int>& field_shape = bottom[1]->shape();
  int batch_size = field_shape[0];
  int field_dim_x = field_shape[1];
  int field_dim_y = field_shape[2];
  int field_dim_z = field_shape[3];
  int field_dim_x_1 = field_dim_x-1;
  int field_dim_y_1 = field_dim_y-1;
  int field_dim_z_1 = field_dim_z-1;
  int num_grids = dim_grid_*dim_grid_*dim_grid_;
  int num_samples = num_grids*num_curve_*len_curve_;

  for (int i = 1; i < bottom.size(); ++i) {
    const Dtype* top_diff = NULL;
    const Dtype* top_normal_diff = NULL;
    if (output_normal_) {
      top_diff = top[2*(i-1)+0]->cpu_diff();
      top_normal_diff = top[2*(i-1)+1]->cpu_diff();
    } else {
      top_diff = top[i-1]->cpu_diff();
    }
    if (propagate_down[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();

      for (int grid_idx = 0; grid_idx < num_grids; ++ grid_idx) {
        int g_offset = grid_idx * num_curve_ * len_curve_;
        for (int c = 0; c < num_curve_; ++ c) {
          int c_offset = c * len_curve_;
          for (int l = 0; l < len_curve_; ++ l) {
            int sample_idx = (g_offset + c_offset + l);
            int p_offset = sample_idx * 4;

            Dtype x = probing_curves[p_offset+0];
            Dtype y = probing_curves[p_offset+1];
            Dtype z = probing_curves[p_offset+2];

            int x0, y0, z0, x1, y1, z1;
            Dtype x_a, y_a, z_a, x_m, y_m, z_m;
            SnapGrid_cpu(x, x0, x1, x_a, x_m, field_dim_x_1);
            SnapGrid_cpu(y, y0, y1, y_a, y_m, field_dim_y_1);
            SnapGrid_cpu(z, z0, z1, z_a, z_m, field_dim_z_1);
            Dtype x_x0 = x-x0;
            Dtype y_y0 = y-y0;
            Dtype z_z0 = z-z0;
            Dtype x1_x = x1-x;
            Dtype y1_y = y1-y;
            Dtype z1_z = z1-z;

            Dtype w_diff_x = 0;
            Dtype w_diff_y = 0;
            Dtype w_diff_z = 0;
            Dtype dx, dy, dz;
            for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
              int top_offset = batch_idx*num_samples + sample_idx;
              const Dtype& t_diff = top_diff[top_offset];

              ComputeGradient_cpu(bottom_data, batch_idx,
                  x0, y0, z0, x1, y1, z1,
                  x_a, y_a, z_a, x_m, y_m, z_m,
                  x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
                  dx, dy, dz, field_dim_x, field_dim_y, field_dim_z);
              w_diff_x += t_diff*dx;
              w_diff_y += t_diff*dy;
              w_diff_z += t_diff*dz;

              if(output_normal_) {
                Dtype nx_dx, nx_dy, nx_dz;
                Dtype ny_dx, ny_dy, ny_dz;
                Dtype nz_dx, nz_dy, nz_dz;
                ComputeNormalGradient_cpu(bottom_data, batch_idx,
                    x, y, z,
                    nx_dx, nx_dy, nx_dz,
                    ny_dx, ny_dy, ny_dz,
                    nz_dx, nz_dy, nz_dz,
                    field_dim_x, field_dim_y, field_dim_z);
                const Dtype& tx_diff = top_normal_diff[3*top_offset+0];
                const Dtype& ty_diff = top_normal_diff[3*top_offset+1];
                const Dtype& tz_diff = top_normal_diff[3*top_offset+2];

                w_diff_x += tx_diff*nx_dx + ty_diff*ny_dx + tz_diff*nz_dx;
                w_diff_y += tx_diff*nx_dy + ty_diff*ny_dy + tz_diff*nz_dy;
                w_diff_z += tx_diff*nx_dz + ty_diff*ny_dz + tz_diff*nz_dz;
              }
            } /* batch_size */

            probing_curves_diff[p_offset+0] += w_diff_x;
            probing_curves_diff[p_offset+1] += w_diff_y;
            probing_curves_diff[p_offset+2] += w_diff_z;
          } /* len_curve */
        } /* num_curve */
      } /* num_grids */
    }
    if (rand()%100 == 0) {
      Dtype amax = caffe_cpu_amax(top[i]->count(), top_diff);
      Dtype aavg = caffe_cpu_aavg(top[i]->count(), top_diff);
      LOG(INFO) << "FieldProbingLayer::Backward_cpu top_diff max-avg: " << amax << "\t" << aavg;
    }
  } /* top.size() */

  if (rand()%100 == 0) {
    Dtype amax = caffe_cpu_amax(bottom[0]->count(), bottom[0]->cpu_diff());
    Dtype aavg = caffe_cpu_aavg(bottom[0]->count(), bottom[0]->cpu_diff());
    LOG(INFO) << "FieldProbingLayer::Backward_cpu probing_curves_diff max-avg: " << amax << "\t" << aavg;
  }
}

#ifdef CPU_ONLY
STUB_GPU(FieldProbingLayer);
#endif

INSTANTIATE_CLASS(FieldProbingLayer);
REGISTER_LAYER_CLASS(FieldProbing);

}  // namespace caffe
