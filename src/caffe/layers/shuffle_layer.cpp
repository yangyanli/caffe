#include "caffe/layers/shuffle_layer.hpp"

namespace caffe {

template <typename Dtype>
void ShuffleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const ShuffleParameter& shuffle_param = this->layer_param_.shuffle_param();
  const int num_axes = shuffle_param.axis_size();
  top_axis_from_bottom_.resize(num_axes);
  for (int i = 0; i < num_axes; ++i) {
    top_axis_from_bottom_[i] = shuffle_param.axis(i);
  }
  bottom_axis_to_top_.resize(num_axes);
  for (int i = 0; i < num_axes; ++i) {
    bottom_axis_to_top_[top_axis_from_bottom_[i]] = i;
  }

  CHECK(bottom[0]->shape().size() == top_axis_from_bottom_.size())
          << "Shuffle axis number should be same as the input axis number.";

  const vector<int>& bottom_shape = bottom[0]->shape();
  output_shape_.resize(num_axes);
  for (int i = 0; i < num_axes; ++i) {
    output_shape_[i] = bottom_shape[top_axis_from_bottom_[i]];
  }
}

template <typename Dtype>
void ShuffleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(output_shape_);
}

static int GetOutputOffsetCPU(int input_offset, const vector<int>& input_shape, const vector<int>& bottom_axis_to_top, const vector<int>& output_shape) {
  int num_axes = input_shape.size();
  vector<int> output_indices(num_axes);
  for (int i = num_axes-1; i >= 0; -- i) {
    output_indices[bottom_axis_to_top[i]] = input_offset%input_shape[i];
    input_offset /= input_shape[i];
  }

  int output_offset = 0;
  for (int i = 0; i < num_axes; ++i) {
    CHECK_GE(output_indices[i], 0);
    CHECK_LT(output_indices[i], output_shape[i]);
    output_offset *= output_shape[i];
    output_offset += output_indices[i];
  }
  return output_offset;
}

template <typename Dtype>
void ShuffleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const vector<int>& bottom_shape = bottom[0]->shape();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[GetOutputOffsetCPU(i, bottom_shape, bottom_axis_to_top_, output_shape_)] = bottom_data[i];
  }
}

template <typename Dtype>
void ShuffleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const vector<int>& bottom_shape = bottom[0]->shape();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[GetOutputOffsetCPU(i, bottom_shape, bottom_axis_to_top_, output_shape_)];
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ShuffleLayer);
#endif

INSTANTIATE_CLASS(ShuffleLayer);
REGISTER_LAYER_CLASS(Shuffle);

}  // namespace caffe
