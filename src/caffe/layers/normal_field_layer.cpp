#include "caffe/util/benchmark.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/normal_field_layer.hpp"

namespace caffe {

template<typename Dtype>
void NormalFieldLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4) << "NormalFieldLayer supports only 3D data.";
  const vector<int>& bottom_shape = bottom[0]->shape();
  bool is_cube = (bottom_shape[1] == bottom_shape[2] && bottom_shape[1] == bottom_shape[3]);
  CHECK_EQ(is_cube, true) << "NormalFieldLayer supports only cube shape data.";
}

template<typename Dtype>
void NormalFieldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape.push_back(3);
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void NormalFieldLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const std::vector<int>& field_shape = bottom[0]->shape();
  int batch_size = field_shape[0];
  int grid_dim = field_shape[1];
  int grid_dim_1 = grid_dim-1;

  std::vector<int> offset(5, 0);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < grid_dim; ++ i) {
    offset[1] = i;
    Dtype x = i + 0.5;
    int x0, x1;
    Dtype x_a, x_m;
    SnapGrid_cpu(x, x0, x1, x_a, x_m, grid_dim_1);
    Dtype x_x0 = x-x0;
    Dtype x1_x = x1-x;
    for (int j = 0; j < grid_dim; ++ j) {
      offset[2] = j;
      Dtype y = j + 0.5;
      int y0, y1;
      Dtype y_a, y_m;
      SnapGrid_cpu(y, y0, y1, y_a, y_m, grid_dim_1);
      Dtype y_y0 = y-y0;
      Dtype y1_y = y1-y;
      for (int k = 0; k < grid_dim; ++ k) {
        offset[3] = k;
        Dtype z = k + 0.5;
        int z0, z1;
        Dtype z_a, z_m;
        SnapGrid_cpu(z, z0, z1, z_a, z_m, grid_dim_1);
        Dtype z_z0 = z-z0;
        Dtype z1_z = z1-z;
        for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) { 
          offset[0] = batch_idx;
          Dtype nx, ny, nz;
          ComputeGradient_cpu(bottom_data, batch_idx, x0, y0, z0, x1, y1, z1,
            x_a, y_a, z_a, x_m, y_m, z_m, x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
            nx, ny, nz, grid_dim, grid_dim, grid_dim);
          Normalize_cpu(nx, ny, nz);
          int p = top[0]->offset(offset); 
          top_data[p+0] = nx;
          top_data[p+1] = ny; 
          top_data[p+2] = nz; 
        } /* batch_size */
      } /* z */
    } /* y */
  } /* x */
}

#ifdef CPU_ONLY
STUB_GPU(NormalFieldLayer);
#endif

INSTANTIATE_CLASS(NormalFieldLayer);
REGISTER_LAYER_CLASS(NormalField);

}  // namespace caffe
