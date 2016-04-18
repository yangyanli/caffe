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
  const vector<int>& field_shape = bottom[0]->shape();
  CHECK(field_shape.size() == 4 || field_shape.size() == 5) << "GradientFieldLayer supports only 4D or 5D data.";
  bool is_cube = (field_shape[1] == field_shape[2] && field_shape[1] == field_shape[3]);
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

  int num_output = field_shape[0]*num_transformations_;
  std::vector<int> transformations_shape;
  transformations_shape.push_back(num_output);
  transformations_shape.push_back(len_transformation_param*1);
  transformations_.Reshape(transformations_shape);

  int field_num = bottom.size()-1;
  output_inverse_transformations_ = (top.size() == field_num+2);
}

template <typename Dtype>
void Transform3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num_output = bottom[0]->shape()[0]*num_transformations_;

  int field_num = bottom.size()-1;
  for (int i = 0; i < field_num; ++ i) {
    std::vector<int> top_shape = bottom[i]->shape();
    top_shape[0] = num_output;
    top[i]->Reshape(top_shape);
  }

  std::vector<int> label_shape(1, num_output);
  top[field_num]->Reshape(label_shape);

  if (output_inverse_transformations_) {
    top[field_num]->Reshape(transformations_.shape());
  }

  Dtype* transformations_data = transformations_.mutable_cpu_data();
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
  int field_num = bottom.size()-1;
  ForwardLabel(bottom[field_num], top[field_num]);
  if (output_inverse_transformations_) {
    ForwardInverseTransformations(&transformations_, top[field_num+1]);
  }

  for (int i = 0; i < field_num; ++ i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    const vector<int>& field_shape = bottom[i]->shape();
    Dtype* top_data = top[i]->mutable_cpu_data();
    const int batch_size = field_shape[0];
    const int grid_dim = field_shape[1];
    const int grid_dim_1 = grid_dim-1;
    const int yz = grid_dim*grid_dim;
    const int num_grids = yz*grid_dim;
    int field_channels = (field_shape.size() == 5)?(field_shape.back()):(1);
    Dtype c_offset = (grid_dim_1)/2.0;
    for (int b_batch_idx = 0; b_batch_idx < batch_size; ++ b_batch_idx) {
      for(int transformation_idx = 0; transformation_idx < num_transformations_; ++ transformation_idx) {
        int t_batch_idx = b_batch_idx * num_transformations_ + transformation_idx;
        const Dtype* t = transformations_.cpu_data() + t_batch_idx*len_transformation_param;
  
        int t_n_offset = t_batch_idx * num_grids;
        for (int x = 0; x < grid_dim; ++ x) {
          Dtype xx = x+0.5-c_offset;
          int t_n_x_offset = t_n_offset + x * yz;
          for (int y = 0; y < grid_dim; ++ y) {
            Dtype yy = y+0.5-c_offset;
            int t_n_x_y_offset = t_n_x_offset + y*grid_dim;
            for (int z = 0; z < grid_dim; ++ z) {
              Dtype zz = z+0.5-c_offset;
  
              Dtype bx = t[0]*xx + t[1]*yy + t[2]*zz + t[3] + c_offset - 0.5;
              Dtype by = t[4]*xx + t[5]*yy + t[6]*zz + t[7] + c_offset - 0.5;
              Dtype bz = t[8]*xx + t[9]*yy + t[10]*zz + t[11] + c_offset - 0.5;
  
              Dtype* t_data = top_data + (t_n_x_y_offset+z)*field_channels;
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
                Interpolate_cpu(bottom_data, b_batch_idx, x0, y0, z0, x1, y1, z1,
                  x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, grid_dim, grid_dim, grid_dim,
                  t_data, field_channels);
              } else {
                caffe_set(field_channels, pad_value_, t_data);
              }
            }
          }
        }
  
      } /* transformations */
    } /* batch_size */
  }
}


#ifdef CPU_ONLY
STUB_GPU(Transform3DLayer);
#endif

INSTANTIATE_CLASS(Transform3DLayer);
REGISTER_LAYER_CLASS(Transform3D);

}  // namespace caffe
