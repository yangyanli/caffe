#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/gradient_field_layer.hpp"

namespace caffe {

template<typename Dtype>
void GradientFieldLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const vector<int>& field_shape = bottom[0]->shape();
  CHECK(field_shape.size() == 4 || field_shape.size() == 5) << "GradientFieldLayer supports only 4D or 5D data.";
  bool is_cube = (field_shape[1] == field_shape[2] && field_shape[1] == field_shape[3]);
  CHECK_EQ(is_cube, true) << "GradientFieldLayer supports only cube shape data.";
}

template<typename Dtype>
void GradientFieldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape = bottom[0]->shape();
  if (top_shape.size() == 5) {
    top_shape.back()*= 3;
  } else {
    top_shape.push_back(3);
  }
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void GradientFieldLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const std::vector<int>& field_shape = bottom[0]->shape();
  int batch_size = field_shape[0];
  int grid_dim = field_shape[1];
  int grid_dim_1 = grid_dim-1;
  int field_channels = (field_shape.size() == 5)?(field_shape.back()):(1);

  std::vector<int> offset(5, 0);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < grid_dim; ++ i) {
    offset[1] = i;
    Dtype x = i + 0.5;
    int x0, x1;
    Dtype x_a, x_m;
    SnapGrid_cpu(x, x0, x1, x_a, x_m, grid_dim_1);
    for (int j = 0; j < grid_dim; ++ j) {
      offset[2] = j;
      Dtype y = j + 0.5;
      int y0, y1;
      Dtype y_a, y_m;
      SnapGrid_cpu(y, y0, y1, y_a, y_m, grid_dim_1);
      for (int k = 0; k < grid_dim; ++ k) {
        offset[3] = k;
        Dtype z = k + 0.5;
        int z0, z1;
        Dtype z_a, z_m;
        SnapGrid_cpu(z, z0, z1, z_a, z_m, grid_dim_1);
        for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) { 
          offset[0] = batch_idx;
          Dtype* t_data = top_data + top[0]->offset(offset);
          ComputeGradient_cpu(bottom_data, batch_idx, x, y, z, x0, y0, z0, x1, y1, z1,
            x_a, y_a, z_a, x_m, y_m, z_m, grid_dim, t_data, field_channels);
          Normalize_cpu(t_data);
        } /* batch_size */
      } /* z */
    } /* y */
  } /* x */
}

#ifdef CPU_ONLY
STUB_GPU(GradientFieldLayer);
#endif

INSTANTIATE_CLASS(GradientFieldLayer);
REGISTER_LAYER_CLASS(GradientField);

}  // namespace caffe
