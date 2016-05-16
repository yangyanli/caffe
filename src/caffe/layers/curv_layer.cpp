#include "caffe/filler.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/curv_layer.hpp"

namespace caffe {

template<typename Dtype>
void CurvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }

  const vector<int>& bottom_shape = bottom[0]->shape();
  batch_size_ = bottom_shape[0];
  num_sliding_ = bottom_shape[1];
  //num_sliding_ = bottom_shape[2];
  //num_sliding_ = bottom_shape[3];
  num_curve_ = bottom_shape[4];
  len_curve_ = bottom_shape[5];
  num_channel_ = bottom_shape[6];

  const CurvolutionParameter& param = this->layer_param_.curvolution_param();
  share_weights_ = param.share_weights();

  vector<int> weight_shape;
  if (!share_weights_) {
    weight_shape.push_back(num_sliding_);
    weight_shape.push_back(num_sliding_);
    weight_shape.push_back(num_sliding_);
  }
  weight_shape.push_back(num_curve_);
  weight_shape.push_back(len_curve_);
  weight_shape.push_back(num_channel_);

  std::vector<int> bias_shape;
  bias_shape.insert(bias_shape.begin(), weight_shape.begin(), weight_shape.end() - 2);

  if (this->blobs_.size() > 0) {
    CHECK_EQ(2, this->blobs_.size()) << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape " << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape " << bias_shaped_blob.shape_string() << "; instead, shape was " << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    // Initialize and fill the weights:
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // Initialize and fill the biases.
    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(param.bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }

#ifndef CPU_ONLY
  if(share_weights_) {
    std::vector<int> slided_weight_shape;
    slided_weight_shape.push_back(num_sliding_);
    slided_weight_shape.push_back(num_sliding_);
    slided_weight_shape.push_back(num_sliding_);
    slided_weight_shape.push_back(num_curve_);
    slided_weight_shape.push_back(len_curve_);
    slided_weight_shape.push_back(num_channel_);
    slided_weight_.Reshape(slided_weight_shape);

    std::vector<int> slided_bias_shape;
    slided_bias_shape.insert(slided_bias_shape.begin(), slided_weight_shape.begin(), slided_weight_shape.end() - 2);
    slided_bias_.Reshape(slided_bias_shape);
  }
#endif // !CPU_ONLY
}

template<typename Dtype>
void CurvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Shape the tops.
  const vector<int>& bottom_shape = bottom[0]->shape();
  vector<int> top_shape;
  top_shape.insert(top_shape.begin(), bottom_shape.begin(), bottom_shape.end() - 2);
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
}

template<typename Dtype>
void CurvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->blobs_[1]->cpu_data();

  int num_sliding_total = num_sliding_*num_sliding_*num_sliding_;
  int num_sliding_curves = num_sliding_total * num_curve_;
  int len_dot = len_curve_ * num_channel_;

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
      int batch_offset = batch_idx * num_sliding_curves;
      for (int sliding_idx = 0; sliding_idx < num_sliding_total; ++ sliding_idx) {
        int sliding_offset = sliding_idx * num_curve_;
        for (int curve_idx = 0; curve_idx < num_curve_; ++curve_idx) {
          int sliding_curve_idx = sliding_offset + curve_idx;
          int top_offset = batch_offset + sliding_curve_idx;
          int bottom_offset = top_offset * len_dot;
          int weight_offset = (share_weights_?(curve_idx):(sliding_curve_idx)) * len_dot;
          top_data[top_offset] = caffe_cpu_dot(len_dot, bottom_data + bottom_offset, weight + weight_offset);
        } /* num_all_curves */
        if(share_weights_) {
          caffe_axpy(num_curve_, (Dtype) (1.), bias, top_data+batch_offset+sliding_offset);
        }
      }
      if(!share_weights_) {
        caffe_axpy(num_sliding_curves, (Dtype) (1.), bias, top_data+batch_offset);
      }
    } /* batch_size */
  } /* bottom.size() */
}

template<typename Dtype>
void CurvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  if (this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  int num_sliding_total = num_sliding_*num_sliding_*num_sliding_;
  int num_sliding_curves = num_sliding_total * num_curve_;
  int len_dot = len_curve_ * num_channel_;

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    // Bias gradient, if necessary.
    if (this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
        int batch_offset = batch_idx * num_sliding_curves;
        if(share_weights_) {
          for (int sliding_idx = 0; sliding_idx < num_sliding_total; ++ sliding_idx) {
            int sliding_offset = sliding_idx * num_curve_;
            caffe_axpy(num_curve_, (Dtype) 1., top_diff + batch_offset + sliding_offset, bias_diff);
          }
        } else {
          caffe_axpy(num_sliding_curves, (Dtype) 1., top_diff + batch_offset, bias_diff);
        }
      } /* batch_size */
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      caffe_set(bottom[i]->count(), Dtype(0), bottom_diff);

      for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
        int batch_offset = batch_idx * num_sliding_curves;
        for (int sliding_idx = 0; sliding_idx < num_sliding_total; ++ sliding_idx) {
          int sliding_offset = sliding_idx * num_curve_;
          for (int curve_idx = 0; curve_idx < num_curve_; ++curve_idx) {
            int sliding_curve_idx = sliding_offset + curve_idx;
            int top_offset = batch_offset + sliding_curve_idx;
            int bottom_offset = top_offset * len_dot;
            int weight_offset = (share_weights_?(curve_idx):(sliding_curve_idx)) * len_dot;
            // gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0]) {
              caffe_axpy(len_dot, top_diff[top_offset], bottom_data + bottom_offset, weight_diff + weight_offset);
            }
            // gradient w.r.t. bottom data, if necessary.
            if (propagate_down[i]) {
              caffe_axpy(len_dot, top_diff[top_offset], weight + weight_offset, bottom_diff + bottom_offset);
            }
          } /* num_curve_ */
        } /* num_sliding_total */
      } /* batch_size */
    }
  } /* top.size() */

  //Dtype scaler = 1.0 / (batch_size_ * top.size());
  Dtype scaler = 1.0 / (top.size());
  caffe_scal(num_sliding_curves * len_dot, scaler, this->blobs_[0]->mutable_cpu_diff());
  caffe_scal(num_sliding_curves, scaler, this->blobs_[1]->mutable_cpu_diff());

  if (rand() % 100 == 0) {
    {
      Dtype amax = caffe_cpu_amax(top[0]->count(), top[0]->cpu_diff());
      Dtype aavg = caffe_cpu_aavg(top[0]->count(), top[0]->cpu_diff());
      LOG(INFO) << "CurvolutionLayer::Backward_cpu top_diff max-avg: " << amax << "\t" << aavg;
    }
    {
      Dtype amax = caffe_cpu_amax(this->blobs_[0]->count(), this->blobs_[0]->cpu_diff());
      Dtype aavg = caffe_cpu_aavg(this->blobs_[0]->count(), this->blobs_[0]->cpu_diff());
      LOG(INFO) << "CurvolutionLayer::Backward_cpu weight_diff max-avg: " << amax << "\t" << aavg;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CurvolutionLayer);
#endif

INSTANTIATE_CLASS(CurvolutionLayer);
REGISTER_LAYER_CLASS(Curvolution);

}  // namespace caffe
