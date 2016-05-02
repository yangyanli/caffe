#define GLM_FORCE_RADIANS
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "caffe/util/rng.hpp"
#include "caffe/util/field_operations.hpp"
#include "caffe/layers/transform_3d_layer.hpp"

namespace caffe {

template <typename Dtype>
const int Transform3DLayer<Dtype>::len_trans_params;

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
  if(order_.empty()) {
    rotations.push_back(r_x);
    rotations.push_back(r_y);
    rotations.push_back(r_z);
    std::random_shuffle(rotations.begin(), rotations.end());
  } else {
    for (int i = 0; i < order_.length(); ++ i) {
      switch (tolower(order_[i])) {
      case 'x':
        rotations.push_back(r_x);
        break;
      case 'y':
        rotations.push_back(r_y);
        break;
      case 'z':
        rotations.push_back(r_z);
        break;
      default:
        break;
      }
    }
  }

  mat4 t = glm::translate(vec3(translation_x, translation_y, translation_z));
  mat4 s = glm::scale(vec3(scaling_x, scaling_y, scaling_z));

  mat4 trans = t * s * rotations[0] * rotations[1] * rotations[2];
  mat4 trans_row_major = glm::transpose(trans);
  const Dtype* p = (const Dtype*)glm::value_ptr(trans_row_major);
  for (int i = 0; i < len_trans_params; ++ i) {
    transformation[i] = p[i];
  }

  /*
  std::cout << std::endl;
  std::cout << transformation[0] << " " << transformation[1] << " " << transformation[2] << " " << transformation[3] << std::endl;
  std::cout << transformation[4] << " " << transformation[5] << " " << transformation[6] << " " << transformation[7] << std::endl;
  std::cout << transformation[8] << " " << transformation[9] << " " << transformation[10] << " " << transformation[11] << std::endl;
  std::cout << std::endl;
  */

  return;
}

template <typename Dtype>
void Transform3DLayer<Dtype>::GetInverseTransformation(const Dtype* transformation, Dtype* inverse_transformation) {
  typedef glm::detail::tmat4x4<Dtype, glm::lowp> mat4;
  mat4 trans_row_major;
  Dtype* p = (Dtype*)glm::value_ptr(trans_row_major);
  for (int i = 0; i < len_trans_params; ++ i) {
    p[i] = transformation[i];
  }
  mat4 trans = glm::transpose(trans_row_major);
  mat4 trans_row_major_inverse = glm::inverseTranspose(trans);

  const Dtype* p_inverse = (const Dtype*)glm::value_ptr(trans_row_major_inverse);
  for (int i = 0; i < len_trans_params; ++ i) {
    inverse_transformation[i] = p_inverse[i];
  }

  return;
}

template <typename Dtype>
void Transform3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const vector<int>& field_shape = bottom[0]->shape();
  CHECK(field_shape.size() == 4 || field_shape.size() == 5) << "GradientFieldLayer supports only 4D or 5D data.";
  bool is_cube = (field_shape[1] == field_shape[2] && field_shape[1] == field_shape[3]);
  CHECK_EQ(is_cube, true) << "Transform3DLayer supports only cube shape data.";

  const Transform3DParameter& param = this->layer_param_.transform_3d_param();
  GetVariateGenerator(rotation_x_, param.min_rotation_x(), param.max_rotation_x());
  GetVariateGenerator(rotation_y_, param.min_rotation_y(), param.max_rotation_y());
  GetVariateGenerator(rotation_z_, param.min_rotation_z(), param.max_rotation_z());
  
  GetVariateGenerator(translation_x_, param.min_translation_x(), param.max_translation_x());
  GetVariateGenerator(translation_y_, param.min_translation_y(), param.max_translation_y());
  GetVariateGenerator(translation_z_, param.min_translation_z(), param.max_translation_z());

  GetVariateGenerator(scaling_x_, param.min_scaling_x(), param.max_scaling_x());
  GetVariateGenerator(scaling_y_, param.min_scaling_y(), param.max_scaling_y());
  GetVariateGenerator(scaling_z_, param.min_scaling_z(), param.max_scaling_z());

  pad_value_ = param.pad_value();
  num_transformations_ = param.num_transformations();
  batch_size_ = field_shape[0];
  order_ = param.order();
  std::reverse(order_.begin(), order_.end());

  int num_output = batch_size_*num_transformations_;
  std::vector<int> transformations_shape;
  transformations_shape.push_back(num_output);
  transformations_shape.push_back(len_trans_params);
  transformations_.Reshape(transformations_shape);
  inverse_transformations_.Reshape(transformations_shape);

  int field_num = bottom.size()-1;
  output_inverse_transformations_ = (top.size() == field_num+2);
}

template <typename Dtype>
void Transform3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num_output = batch_size_*num_transformations_;

  int field_num = bottom.size()-1;
  for (int i = 0; i < field_num; ++ i) {
    std::vector<int> top_shape = bottom[i]->shape();
    top_shape[0] = num_output;
    top[i]->Reshape(top_shape);
  }

  std::vector<int> label_shape(1, num_output);
  top[field_num]->Reshape(label_shape);

  if (output_inverse_transformations_) {
    top[field_num+1]->Reshape(transformations_.shape());
  }

  Dtype* transformations_data = transformations_.mutable_cpu_data();
  Dtype* inverse_transformations_data = inverse_transformations_.mutable_cpu_data();
  for (int i = 0; i < num_output; ++ i) {
    int offset = i*len_trans_params;
    GetTransformation(transformations_data+offset);
    GetInverseTransformation(transformations_data+offset, inverse_transformations_data+offset);
  }
}


template <typename Dtype>
void Transform3DLayer<Dtype>::ForwardLabel(Blob<Dtype>* input_labels, Blob<Dtype>* output_labels) {
  const Dtype* input_data = input_labels->cpu_data();
  Dtype* output_data = output_labels->mutable_cpu_data();
  for (int i = 0; i < batch_size_; ++ i) {
    int offset = i*num_transformations_;
    for (int j = 0; j < num_transformations_; ++ j) {
      output_data[offset+j] = input_data[i];
    }
  }
}

template <typename Dtype>
void Transform3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int field_num = bottom.size()-1;
  ForwardLabel(bottom[field_num], top[field_num]);

  int num_output = batch_size_*num_transformations_;
  if (output_inverse_transformations_) {
    caffe_copy(num_output*len_trans_params, inverse_transformations_.cpu_data(), top[field_num+1]->mutable_cpu_data());
  }

  for (int i = 0; i < field_num; ++ i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    const vector<int>& field_shape = bottom[i]->shape();
    Dtype* top_data = top[i]->mutable_cpu_data();
    const int field_dim = field_shape[1];
    const int field_dim_1 = field_dim-1;
    const int yz = field_dim*field_dim;
    const int num_grids = yz*field_dim;
    int field_channels = (field_shape.size() == 5)?(field_shape.back()):(1);
    Dtype c_offset = field_dim_1/2.0;
    for (int b_batch_idx = 0; b_batch_idx < batch_size_; ++ b_batch_idx) {
      for(int transformation_idx = 0; transformation_idx < num_transformations_; ++ transformation_idx) {
        int t_batch_idx = b_batch_idx * num_transformations_ + transformation_idx;
        const Dtype* t = transformations_.cpu_data() + t_batch_idx*len_trans_params;
  
        int t_n_offset = t_batch_idx * num_grids;
        for (int x = 0; x < field_dim; ++ x) {
          Dtype xx = x-c_offset;
          int t_n_x_offset = t_n_offset + x * yz;
          for (int y = 0; y < field_dim; ++ y) {
            Dtype yy = y-c_offset;
            int t_n_x_y_offset = t_n_x_offset + y*field_dim;
            for (int z = 0; z < field_dim; ++ z) {
              Dtype zz = z-c_offset;
  
              Dtype bx = t[0]*xx + t[1]*yy + t[2]*zz + t[3] + c_offset;
              Dtype by = t[4]*xx + t[5]*yy + t[6]*zz + t[7] + c_offset;
              Dtype bz = t[8]*xx + t[9]*yy + t[10]*zz + t[11] + c_offset;
  
              Dtype* t_data = top_data + (t_n_x_y_offset+z)*field_channels;
              if(bx >= 0 && bx < field_dim
                  && by >= 0 && by < field_dim
                  && bz >= 0 && bz < field_dim) {
                int x0, y0, z0, x1, y1, z1;
                SnapGrid_cpu(bx, x0, x1, field_dim_1);
                SnapGrid_cpu(by, y0, y1, field_dim_1);
                SnapGrid_cpu(bz, z0, z1, field_dim_1);
                Interpolate_cpu(bottom_data, b_batch_idx, bx, by, bz, x0, y0, z0, x1, y1, z1,
                  field_dim, t_data, field_channels);
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
