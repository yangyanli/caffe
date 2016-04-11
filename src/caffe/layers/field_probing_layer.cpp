#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/field_probing_layer.hpp"

namespace caffe {

template<typename Dtype>
void FieldProbingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const std::vector<int>& probing_curves_shape = bottom[0]->shape();
  CHECK_EQ(probing_curves_shape.size(), 7) << "Probing curves must be in N*D*D*D*C*L*4 shape.";

  int batch_size = bottom[1]->shape()[0];
  CHECK_EQ(probing_curves_shape[0], batch_size) << "Probing curves must have the same batch size as input field(s).";

  for (int bottom_id = 2; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[1]->shape() == bottom[bottom_id]->shape())
        << "All input fields must have the same shape.";
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const vector<int>& probing_curves_shape = bottom[0]->shape();
  vector<int> top_shape = probing_curves_shape;
  top_shape.back() = 1;

  for (int i = 1; i < bottom.size(); ++ i) {
    top[i-1]->Reshape(top_shape);
  }
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const vector<int>& field_shape = bottom[1]->shape();
  int batch_size = field_shape[0];
  int field_dim_x = field_shape[1];
  int field_dim_y = field_shape[2];
  int field_dim_z = field_shape[3];
  int field_dim_x_1 = field_dim_x-1;
  int field_dim_y_1 = field_dim_y-1;
  int field_dim_z_1 = field_dim_z-1;

  int probing_curves_size = bottom[0]->count(1);
  int num_samples = probing_curves_size/4;

  for (int i = 1; i < bottom.size(); ++ i) {
    const Dtype* probing_curves = bottom[0]->cpu_data()+probing_curves_size*(i-1);
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i-1]->mutable_cpu_data();

    for (int sample_idx = 0; sample_idx < num_samples; ++ sample_idx) {
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
        int top_offset = n_offset + sample_idx;
        top_data[top_offset] = Interpolate_cpu(bottom_data, batch_idx,
            x0, y0, z0, x1, y1, z1, x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
            field_dim_x, field_dim_y, field_dim_z);
      } /* batch_size */
    } /* num_samples */
  } /* bottom.size() */
}

template<typename Dtype>
void FieldProbingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());

  const vector<int>& field_shape = bottom[1]->shape();
  int batch_size = field_shape[0];
  int field_dim_x = field_shape[1];
  int field_dim_y = field_shape[2];
  int field_dim_z = field_shape[3];
  int field_dim_x_1 = field_dim_x-1;
  int field_dim_y_1 = field_dim_y-1;
  int field_dim_z_1 = field_dim_z-1;
  int probing_curves_size = bottom[0]->count(1);
  int num_samples = probing_curves_size/4;

  for (int i = 1; i < bottom.size(); ++i) {
    if (!(propagate_down[0] || propagate_down[i])) 
      continue;

    const Dtype* probing_curves = bottom[0]->cpu_data()+probing_curves_size*(i-1);
    Dtype* probing_curves_diff = bottom[0]->mutable_cpu_diff()+probing_curves_size*(i-1);
    const Dtype* top_diff = top[i-1]->cpu_diff();
 
    const Dtype* bottom_data = bottom[i]->cpu_data();
    for (int sample_idx = 0; sample_idx < num_samples; ++ sample_idx) {
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
      } /* batch_size */

      probing_curves_diff[p_offset+0] += w_diff_x;
      probing_curves_diff[p_offset+1] += w_diff_y;
      probing_curves_diff[p_offset+2] += w_diff_z;
    } /* num_samples */

    if (rand()%100 == 0) {
      Dtype amax = caffe_cpu_amax(top[i-1]->count(), top_diff);
      Dtype aavg = caffe_cpu_aavg(top[i-1]->count(), top_diff);
      LOG(INFO) << "FieldProbingLayer::Backward_cpu top_diff max-avg: " << amax << "\t" << aavg;
    }
  } /* bottom.size() */

  caffe_scal(bottom[0]->count(), Dtype(1.0/(bottom.size()-1)), bottom[0]->mutable_cpu_diff());
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
