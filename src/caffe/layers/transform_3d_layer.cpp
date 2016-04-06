#define GLM_FORCE_RADIANS
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "caffe/util/rng.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/transform_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
void Transform3DLayer<Dtype>::GetVariateGenerator(boost::shared_ptr<VariateGenerator>& vg, Dtype min, Dtype max)
{
  if (max == min) {
    max = boost::math::float_next(min);
  }
  boost::uniform_real<Dtype> uniform_distribution(min, max);
  vg.reset(new VariateGenerator(caffe_rng(), uniform_distribution));
}

template <typename Dtype>
void Transform3DLayer<Dtype>::GetTransformation(Dtype* transformation) {
  Dtype scaler = M_PI/180.0;
  Dtype rotation_x = (*rotation_x_)()*scaler;
  Dtype rotation_y = (*rotation_y_)()*scaler;
  Dtype rotation_z = (*rotation_z_)()*scaler;
  
  Dtype translation_x = (*translation_x_)();
  Dtype translation_y = (*translation_y_)();
  Dtype translation_z = (*translation_z_)();

  Dtype scaling_x = (*scaling_x_)();
  Dtype scaling_y = (*scaling_y_)();
  Dtype scaling_z = (*scaling_z_)();

  typedef glm::detail::tmat4x4<Dtype, glm::lowp> mat4;
  typedef glm::detail::tvec3<Dtype, glm::lowp> vec3;
  mat4 r_x = glm::rotate(rotation_x, vec3(1.0f, 0.0f, 0.0f));
  mat4 r_y = glm::rotate(rotation_y, vec3(0.0f, 1.0f, 0.0f));
  mat4 r_z = glm::rotate(rotation_z, vec3(0.0f, 0.0f, 1.0f));
  std::vector<mat4> rotations;
  rotations.push_back(r_x);
  rotations.push_back(r_y);
  rotations.push_back(r_z);
  std::random_shuffle(rotations.begin(), rotations.end());

  mat4 t = glm::translate(vec3(translation_x, translation_y, translation_z));
  mat4 s = glm::scale(vec3(scaling_x, scaling_y, scaling_z));

  mat4 trans = t * s * rotations[0] * rotations[1] * rotations[2];
  mat4 trans_row_major = glm::transpose(trans);
  const Dtype* p = (const Dtype*)glm::value_ptr(trans_row_major);
  for (int i = 0; i < len_transformation_param; ++ i) {
    transformation[i] = p[i];
  }  

  return;
}

template <typename Dtype>
void Transform3DLayer<Dtype>::GetInverseTransformation(const Dtype* transformation, Dtype* inverse_transformation) {
  typedef glm::detail::tmat4x4<Dtype, glm::lowp> mat4;
  mat4 trans_row_major;
  Dtype* p = (Dtype*)glm::value_ptr(trans_row_major);
  for (int i = 0; i < len_transformation_param; ++ i) {
    p[i] = transformation[i];
  }
  mat4 trans = glm::transpose(trans_row_major);
  mat4 trans_row_major_inverse = glm::inverseTranspose(trans);

  const Dtype* p_inverse = (const Dtype*)glm::value_ptr(trans_row_major_inverse);
  for (int i = 0; i < len_transformation_param; ++ i) {
    inverse_transformation[i] = p_inverse[i];
  }
  inverse_transformation[0] -= 1;
  inverse_transformation[5] -= 1;
  inverse_transformation[10] -= 1;

  return;
}

template <typename Dtype>
void Transform3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4) << "Transform3DLayer supports only 3D data.";
  const vector<int>& bottom_shape = bottom[0]->shape();
  bool is_cube = (bottom_shape[1] == bottom_shape[2] && bottom_shape[1] == bottom_shape[3]);
  CHECK_EQ(is_cube, true) << "Transform3DLayer supports only cube shape data.";

  const Transform3DParameter& transform_3d_param = this->layer_param_.transform_3d_param();
  GetVariateGenerator(rotation_x_, transform_3d_param.min_rotation_x(), transform_3d_param.max_rotation_x());
  GetVariateGenerator(rotation_y_, transform_3d_param.min_rotation_y(), transform_3d_param.max_rotation_y());
  GetVariateGenerator(rotation_z_, transform_3d_param.min_rotation_z(), transform_3d_param.max_rotation_z());
  
  GetVariateGenerator(translation_x_, transform_3d_param.min_translation_x(), transform_3d_param.max_translation_x());
  GetVariateGenerator(translation_y_, transform_3d_param.min_translation_y(), transform_3d_param.max_translation_y());
  GetVariateGenerator(translation_z_, transform_3d_param.min_translation_z(), transform_3d_param.max_translation_z());

  GetVariateGenerator(scaling_x_, transform_3d_param.min_scaling_x(), transform_3d_param.max_scaling_x());
  GetVariateGenerator(scaling_y_, transform_3d_param.min_scaling_y(), transform_3d_param.max_scaling_y());
  GetVariateGenerator(scaling_z_, transform_3d_param.min_scaling_z(), transform_3d_param.max_scaling_z());

  pad_value_ = transform_3d_param.pad_value();
  num_transformations_ = transform_3d_param.num_transformations();

  int num_output = bottom_shape[0]*num_transformations_;
  std::vector<int> transformations_shape;
  transformations_shape.push_back(num_output);
  transformations_shape.push_back(len_transformation_param*1);
  transformations_.Reshape(transformations_shape);

  output_inverse_transformations_ = (top.size() == 3);
}

template <typename Dtype>
void Transform3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> top_0_shape = bottom[0]->shape();
  top_0_shape[0] *= num_transformations_;
  top[0]->Reshape(top_0_shape);

  vector<int> top_1_shape = bottom[1]->shape();
  top_1_shape[0] = top_0_shape[0];
  top[1]->Reshape(top_1_shape);

  top[2]->Reshape(transformations_.shape());

  Dtype* transformations_data = transformations_.mutable_cpu_data();
  int num_output = top_0_shape[0];
  for (int i = 0; i < num_output; ++ i) {
    GetTransformation(transformations_data+i*len_transformation_param);
  }
}


template <typename Dtype>
void Transform3DLayer<Dtype>::ForwardLabel(Blob<Dtype>* input_labels, Blob<Dtype>* output_labels) {
  const Dtype* input_data = input_labels->cpu_data();
  Dtype* output_data = output_labels->mutable_cpu_data();
  int num_input = input_labels->count();
  for (int i = 0; i < num_input; ++ i) {
    int offset = i*num_transformations_;
    for (int j = 0; j < num_transformations_; ++ j) {
      output_data[offset+j] = input_data[i];
    }
  }
}

template <typename Dtype>
void Transform3DLayer<Dtype>::ForwardInverseTransformations(Blob<Dtype>* transformations, Blob<Dtype>* inverse_transformations) {
  int num_output = transformations->count()/len_transformation_param;
  for (int i = 0; i < num_output; ++ i) {
    int offset = i*len_transformation_param;
    GetInverseTransformation(transformations->cpu_data()+offset, inverse_transformations->mutable_cpu_data()+offset);
  }
}

template <typename Dtype>
void Transform3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ForwardLabel(bottom[1], top[1]);
  if (output_inverse_transformations_) {
    ForwardInverseTransformations(&transformations_, top[1]);
  }

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const vector<int>& bottom_shape = bottom[0]->shape();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int batch_size = bottom_shape[0];
  const int grid_dim = bottom_shape[1];
  const int grid_dim_1 = grid_dim-1;
  const int yz = grid_dim*grid_dim;
  const int num_grids = bottom[0]->count(1);
  Dtype c_offset = (grid_dim_1)/2.0;
  const Dtype* transformations_data = transformations_.cpu_data();
  for (int b_batch_idx = 0; b_batch_idx < batch_size; ++ b_batch_idx) {
    for(int transformation_idx = 0; transformation_idx < num_transformations_; ++ transformation_idx) {
      int t_batch_idx = b_batch_idx * num_transformations_ + transformation_idx;

      int p = t_batch_idx*len_transformation_param;
      Dtype a = transformations_data[p++];
      Dtype b = transformations_data[p++];
      Dtype c = transformations_data[p++];
      Dtype tx = transformations_data[p++];
      Dtype d = transformations_data[p++];
      Dtype e = transformations_data[p++];
      Dtype f = transformations_data[p++];
      Dtype ty = transformations_data[p++];
      Dtype g = transformations_data[p++];
      Dtype h = transformations_data[p++];
      Dtype i = transformations_data[p++];
      Dtype tz = transformations_data[p++];

      int t_n_offset = t_batch_idx * num_grids;
      for (int x = 0; x < grid_dim; ++ x) {
        Dtype xx = x+0.5-c_offset;
        int t_n_x_offset = t_n_offset + x * yz;
        for (int y = 0; y < grid_dim; ++ y) {
          Dtype yy = y+0.5-c_offset;
          int t_n_x_y_offset = t_n_x_offset + y*grid_dim;
          for (int z = 0; z < grid_dim; ++ z) {
            Dtype zz = z+0.5-c_offset;

            Dtype bx = a*xx + b*yy + c*zz + tx + c_offset;
            Dtype by = d*xx + e*yy + f*zz + ty + c_offset;
            Dtype bz = g*xx + h*yy + i*zz + tz + c_offset;

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
              top_data[t_n_x_y_offset+z] = Interpolate_cpu(bottom_data, b_batch_idx,
                  x0, y0, z0, x1, y1, z1,
                  x_x0, y_y0, z_z0, x1_x, y1_y, z1_z,
                  grid_dim, grid_dim, grid_dim) ;
            } else {
              top_data[t_n_x_y_offset+z] = pad_value_;
            }
          }
        }
      }

    } /* transformations */
  } /* batch_size */
}


#ifdef CPU_ONLY
STUB_GPU(Transform3DLayer);
#endif

INSTANTIATE_CLASS(Transform3DLayer);
REGISTER_LAYER_CLASS(Transform3D);

}  // namespace caffe
