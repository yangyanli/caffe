#include <vector>

#include <boost/random.hpp>

#include "caffe/util/rng.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/rotate_layer.hpp"

namespace caffe {

template <typename Dtype>
void RotateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4) << "RotateLayer supports only 3D data.";
  const vector<int>& bottom_shape = bottom[0]->shape();
  bool is_cube = (bottom_shape[1] == bottom_shape[2] && bottom_shape[1] == bottom_shape[3]);
  CHECK_EQ(is_cube, true) << "RotateLayer supports only cube shape data.";

  const RotateParameter& rotate_param = this->layer_param_.rotate_param();
  min_rotation_x_ = rotate_param.min_rotation_x();
  min_rotation_y_ = rotate_param.min_rotation_y();
  min_rotation_z_ = rotate_param.min_rotation_z();
  max_rotation_x_ = rotate_param.max_rotation_x();
  max_rotation_y_ = rotate_param.max_rotation_y();
  max_rotation_z_ = rotate_param.max_rotation_z();
  num_rotation_ = rotate_param.num_rotation();
  pad_value_ = rotate_param.pad_value();

  int num_output = bottom_shape[0]*num_rotation_;
  vector<int> rotations_shape;
  rotations_shape.push_back(num_output*9);
  rotations_.Reshape(rotations_shape);
}

template <typename Dtype>
void RotateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> top_0_shape = bottom[0]->shape();
  top_0_shape[0] *= num_rotation_;
  top[0]->Reshape(top_0_shape);

  vector<int> top_1_shape = bottom[1]->shape();
  top_1_shape[0] = top_0_shape[0];
  top[1]->Reshape(top_1_shape);

  // Prepare rotations
  typedef boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> > VariateGenerator;
  boost::uniform_real<Dtype> uniform_distribution_x(min_rotation_x_, max_rotation_x_);
  VariateGenerator rand_x(caffe_rng(), uniform_distribution_x);

  boost::uniform_real<Dtype> uniform_distribution_y(min_rotation_y_, max_rotation_y_);
  VariateGenerator rand_y(caffe_rng(), uniform_distribution_y);

  boost::uniform_real<Dtype> uniform_distribution_z(min_rotation_z_, max_rotation_z_);
  VariateGenerator rand_z(caffe_rng(), uniform_distribution_z);

  Dtype scaler = M_PI/180.0;
  Dtype* rotations_data = rotations_.mutable_cpu_data();
  int num_output = top_0_shape[0];
  for (int i = 0; i < num_output; ++ i) {
    Dtype rotation_x = (min_rotation_x_ == max_rotation_x_)?(0):(rand_x()*scaler);
    Dtype rotation_y = (min_rotation_y_ == max_rotation_y_)?(0):(rand_y()*scaler);
    Dtype rotation_z = (min_rotation_z_ == max_rotation_z_)?(0):(rand_z()*scaler);
    Dtype c1 = cos(rotation_x);
    Dtype s1 = sin(rotation_x);
    Dtype c2 = cos(rotation_y);
    Dtype s2 = sin(rotation_y);
    Dtype c3 = cos(rotation_z);
    Dtype s3 = sin(rotation_z);

    // https://en.wikipedia.org/wiki/Euler_angles
    // XYZ
    //Dtype rotation[3][3] = {
    //    {c2*c3, -c2*s3, s2},
    //    {c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1},
    //    {s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2}
    //};
    // XZY
    Dtype rotation[3][3] = {
        {c2*c3, -s2, c2*s3},
        {s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1},
        {c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3}
    };

    rotations_data[i*9+0] = rotation[0][0];
    rotations_data[i*9+1] = rotation[0][1];
    rotations_data[i*9+2] = rotation[0][2];
    rotations_data[i*9+3] = rotation[1][0];
    rotations_data[i*9+4] = rotation[1][1];
    rotations_data[i*9+5] = rotation[1][2];
    rotations_data[i*9+6] = rotation[2][0];
    rotations_data[i*9+7] = rotation[2][1];
    rotations_data[i*9+8] = rotation[2][2];
  }
}


template <typename Dtype>
void RotateLayer<Dtype>::ForwardLabel(Blob<Dtype>* input_labels, Blob<Dtype>* output_labels) {
  const Dtype* input_data = input_labels->cpu_data();
  Dtype* output_data = output_labels->mutable_cpu_data();
  int num_input = input_labels->count();
  for (int i = 0; i < num_input; ++ i) {
    int offset = i*num_rotation_;
    for (int j = 0; j < num_rotation_; ++ j) {
      output_data[offset+j] = input_data[i];
    }
  }
}

template <typename Dtype>
void RotateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ForwardLabel(bottom[1], top[1]);

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const vector<int>& bottom_shape = bottom[0]->shape();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int batch_size = bottom_shape[0];
  const int grid_dim = bottom_shape[1];
  const int grid_dim_1 = grid_dim-1;
  const int yz = grid_dim*grid_dim;
  const int num_grids = bottom[0]->count(1);
  Dtype c_offset = (grid_dim_1)/2.0;
  const Dtype* rotations_data = rotations_.cpu_data();
  for (int b_batch_idx = 0; b_batch_idx < batch_size; ++ b_batch_idx) {
    for(int rotation_idx = 0; rotation_idx < num_rotation_; ++ rotation_idx) {
      int t_batch_idx = b_batch_idx * num_rotation_ + rotation_idx;

      int r_offset = t_batch_idx*9;
      Dtype r00 = rotations_data[r_offset++];
      Dtype r01 = rotations_data[r_offset++];
      Dtype r02 = rotations_data[r_offset++];
      Dtype r10 = rotations_data[r_offset++];
      Dtype r11 = rotations_data[r_offset++];
      Dtype r12 = rotations_data[r_offset++];
      Dtype r20 = rotations_data[r_offset++];
      Dtype r21 = rotations_data[r_offset++];
      Dtype r22 = rotations_data[r_offset++];

      int t_n_offset = t_batch_idx * num_grids;
      for (int tx = 0; tx < grid_dim; ++ tx) {
        Dtype txx = tx+0.5-c_offset;
        int t_n_x_offset = t_n_offset + tx * yz;
        for (int ty = 0; ty < grid_dim; ++ ty) {
          Dtype tyy = ty+0.5-c_offset;
          int t_n_x_y_offset = t_n_x_offset + ty*grid_dim;
          for (int tz = 0; tz < grid_dim; ++ tz) {
            Dtype tzz = tz+0.5-c_offset;

            Dtype bx = r00*txx + r01*tyy + r02*tzz + c_offset;
            Dtype by = r10*txx + r11*tyy + r12*tzz + c_offset;
            Dtype bz = r20*txx + r21*tyy + r22*tzz + c_offset;

            if(bx >= 0 && bx < grid_dim
                && by >= 0 && by < grid_dim
                && bz >= 0 && bz < grid_dim) {
              int x0, y0, z0, x1, y1, z1;
              SnapGrid_cpu(bx, x0, x1, grid_dim_1);
              SnapGrid_cpu(by, y0, y1, grid_dim_1);
              SnapGrid_cpu(bz, z0, z1, grid_dim_1);
              Dtype x_x0 = bx-x0;
              Dtype y_y0 = by-y0;
              Dtype z_z0 = bz-z0;
              Dtype x1_x = x1-bx;
              Dtype y1_y = y1-by;
              Dtype z1_z = z1-bz;
              top_data[t_n_x_y_offset+tz] = Interpolate_cpu(bottom_data, b_batch_idx,
                  x0, y0, z0, x1, y1, z1,
                  x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
                  grid_dim, grid_dim, grid_dim) ;
            } else {
              top_data[t_n_x_y_offset+tz] = pad_value_;
            }
          }
        }
      }

    } /* rotations */
  } /* batch_size */
}


#ifdef CPU_ONLY
STUB_GPU(RotateLayer);
#endif

INSTANTIATE_CLASS(RotateLayer);
REGISTER_LAYER_CLASS(Rotate);

}  // namespace caffe
