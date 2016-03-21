#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_4d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void Pooling4DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 5) << "Pooling4DLayer supports only 4D data.";
  const vector<int>& bottom_shape = bottom[0]->shape();
  bool is_cube = (bottom_shape[1] == bottom_shape[2] && bottom_shape[1] == bottom_shape[3]);
  CHECK_EQ(is_cube, true) << "Pooling4DLayer supports only cube shape data.";

  const Pooling4DParameter& pool_4d_param = this->layer_param_.pooling_4d_param();
  kernel_s_ = pool_4d_param.kernel_s();
  kernel_c_ = pool_4d_param.kernel_c();
  stride_s_ = pool_4d_param.stride_s();
  stride_c_ = pool_4d_param.stride_c();
  pad_s_ = pool_4d_param.pad_s();
  pad_c_ = pool_4d_param.pad_c();
}

template <typename Dtype>
void Pooling4DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const vector<int>& bottom_shape = bottom[0]->shape();
  spatial_ = bottom_shape[1];
  channel_ = bottom_shape[4];
  pooled_spatial_ = static_cast<int>(ceil(static_cast<float>(spatial_ + 2 * pad_s_ - kernel_s_) / stride_s_)) + 1;
  pooled_channel_ = static_cast<int>(ceil(static_cast<float>(channel_ + 2 * pad_c_ - kernel_c_) / stride_c_)) + 1;
  vector<int> top_shape;
  top_shape.push_back(bottom_shape[0]);
  top_shape.push_back(pooled_spatial_);
  top_shape.push_back(pooled_spatial_);
  top_shape.push_back(pooled_spatial_);
  top_shape.push_back(pooled_channel_);
  top[0]->Reshape(top_shape);  
  if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
    max_idx_.Reshape(top_shape);
  }
}

template <typename Dtype>
void Pooling4DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int batch_size = bottom[0]->shape(0);
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX: {
    unsigned int* mask = max_idx_.mutable_cpu_data();
    for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
      for (int x = 0; x < pooled_spatial_; ++ x) {
        int xs = x*stride_s_-pad_s_;
        int xe = min(xs+kernel_s_, spatial_);
        xs = max(xs, 0);
        for (int y = 0; y < pooled_spatial_; ++ y) {
          int ys = y*stride_s_-pad_s_;
          int ye = min(ys+kernel_s_, spatial_);
          ys = max(ys, 0);
          for (int z = 0; z < pooled_spatial_; ++ z) {
            int zs = z*stride_s_-pad_s_;
            int ze = min(zs+kernel_s_, spatial_);
            zs = max(zs, 0);
            for (int c = 0; c < pooled_channel_; ++ c) {
              int cs = c*stride_c_-pad_c_;
              int ce = min(cs+kernel_c_, channel_);
              cs = max(cs, 0);

              int top_offset = (((batch_idx*pooled_spatial_+x)*pooled_spatial_+y)*pooled_spatial_+z)*pooled_channel_+c;
              int max_idx = 0;
              Dtype max_value = -FLT_MAX;
              for (int xx = xs; xx < xe; ++ xx) {
                for (int yy = ys; yy < ye; ++ yy) {
                  for (int zz = zs; zz < ze; ++ zz) {
                    for (int cc = cs; cc < ce; ++ cc) {
                      int bottom_offset = (((batch_idx*spatial_+xx)*spatial_+yy)*spatial_+zz)*channel_+cc;
                      const Dtype& bottom_value = bottom_data[bottom_offset];
                      if (bottom_value > max_value) {
                        max_value = bottom_value;
                        max_idx = bottom_offset;
                      }
                    }
                  }
                }
              }
              mask[top_offset] = max_idx;
              top_data[top_offset] = max_value;
            }
          }
        }
      }
    }
    break;
  } case PoolingParameter_PoolMethod_AVE:
    for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
      for (int x = 0; x < pooled_spatial_; ++ x) {
        int xs = x*stride_s_-pad_s_;
        int xe = min(xs+kernel_s_, spatial_+pad_s_);
        int pool_size_x = xe-xs;
        xs = max(xs, 0);
        xe = min(xe, spatial_);
        for (int y = 0; y < pooled_spatial_; ++ y) {
          int ys = y*stride_s_-pad_s_;
          int ye = min(ys+kernel_s_, spatial_+pad_s_);
          int pool_size_y = ye-ys;
          ys = max(ys, 0);
          ye = min(ye, spatial_);
          for (int z = 0; z < pooled_spatial_; ++ z) {
            int zs = z*stride_s_-pad_s_;
            int ze = min(zs+kernel_s_, spatial_+pad_s_);
            int pool_size_z = ze-zs;
            zs = max(zs, 0);
            ze = min(ze, spatial_);
            for (int c = 0; c < pooled_channel_; ++ c) {
              int cs = c*stride_c_-pad_c_;
              int ce = min(cs+kernel_c_, channel_+pad_c_);
              int pool_size_c = ce-cs;
              cs = max(cs, 0);
              ce = min(ce, channel_);

              int top_offset = (((batch_idx*pooled_spatial_+x)*pooled_spatial_+y)*pooled_spatial_+z)*pooled_channel_+c;
              Dtype pool_size = pool_size_x*pool_size_y*pool_size_z*pool_size_c;
              Dtype avg_value = 0;
              for (int xx = xs; xx < xe; ++ xx) {
                for (int yy = ys; yy < ye; ++ yy) {
                  for (int zz = zs; zz < ze; ++ zz) {
                    for (int cc = cs; cc < ce; ++ cc) {
                      int bottom_offset = (((batch_idx*spatial_+xx)*spatial_+yy)*spatial_+zz)*channel_+cc;
                      avg_value += bottom_data[bottom_offset];
                    }
                  }
                }
              }
              top_data[top_offset] = avg_value/pool_size;
            }
          }
        }
      }
    }

    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void Pooling4DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const int batch_size = bottom[0]->shape(0);

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX: {
    const unsigned int* mask = max_idx_.cpu_data();
    for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
      for (int x = 0; x < pooled_spatial_; ++ x) {
        for (int y = 0; y < pooled_spatial_; ++ y) {
          for (int z = 0; z < pooled_spatial_; ++ z) {
            for (int c = 0; c < pooled_channel_; ++ c) {
              int top_offset = (((batch_idx*pooled_spatial_+x)*pooled_spatial_+y)*pooled_spatial_+z)*pooled_channel_+c;
              bottom_diff[mask[top_offset]] += top_diff[top_offset];
            }
          }
        }
      }
    }
    break;
  } case PoolingParameter_PoolMethod_AVE:
    for (int batch_idx = 0; batch_idx < batch_size; ++ batch_idx) {
      for (int x = 0; x < pooled_spatial_; ++ x) {
        int xs = x*stride_s_-pad_s_;
        int xe = min(xs+kernel_s_, spatial_+pad_s_);
        int pool_size_x = xe-xs;
        xs = max(xs, 0);
        xe = min(xe, spatial_);
        for (int y = 0; y < pooled_spatial_; ++ y) {
          int ys = y*stride_s_-pad_s_;
          int ye = min(ys+kernel_s_, spatial_+pad_s_);
          int pool_size_y = ye-ys;
          ys = max(ys, 0);
          ye = min(ye, spatial_);
          for (int z = 0; z < pooled_spatial_; ++ z) {
            int zs = z*stride_s_-pad_s_;
            int ze = min(zs+kernel_s_, spatial_+pad_s_);
            int pool_size_z = ze-zs;
            zs = max(zs, 0);
            ze = min(ze, spatial_);
            for (int c = 0; c < pooled_channel_; ++ c) {
              int cs = c*stride_c_-pad_c_;
              int ce = min(cs+kernel_c_, channel_+pad_c_);
              int pool_size_c = ce-cs;
              cs = max(cs, 0);
              ce = min(ce, spatial_);

              int top_offset = (((batch_idx*pooled_spatial_+x)*pooled_spatial_+y)*pooled_spatial_+z)*pooled_channel_+c;
              Dtype pool_size = pool_size_x*pool_size_y*pool_size_z*pool_size_c;
              Dtype diff = top_diff[top_offset]/pool_size;
              for (int xx = xs; xx < xe; ++ xx) {
                for (int yy = ys; yy < ye; ++ yy) {
                  for (int zz = zs; zz < ze; ++ zz) {
                    for (int cc = cs; cc < ce; ++ cc) {
                      int bottom_offset = (((batch_idx*spatial_+xx)*spatial_+yy)*spatial_+zz)*channel_+cc;
                      bottom_diff[bottom_offset] += diff;
                    }
                  }
                }
              }
              // end of inner loops
            }
          }
        }
      }
    }
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(Pooling4DLayer);
#endif

INSTANTIATE_CLASS(Pooling4DLayer);
REGISTER_LAYER_CLASS(Pooling4D);

}  // namespace caffe
