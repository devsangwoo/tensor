/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/contrib/lite/kernels/internal/test_util.h"

#include <cmath>
#include <iterator>

namespace tflite {

Dims<4> MakeDimsForInference(int depth, int width, int height, int batch) {
  Dims<4> result;
  int cum_prod = 1;

  result.sizes[0] = depth;
  result.strides[0] = cum_prod;
  cum_prod *= result.sizes[0];

  result.sizes[1] = width;
  result.strides[1] = cum_prod;
  cum_prod *= result.sizes[1];

  result.sizes[2] = height;
  result.strides[2] = cum_prod;
  cum_prod *= result.sizes[2];

  result.sizes[3] = batch;
  result.strides[3] = cum_prod;

  return result;
}

// this is a copied from an internal function in propagate_fixed_sizes.cc
bool ComputeConvSizes(Dims<4> input_dims, int output_depth, int filter_width,
                      int filter_height, int stride, int dilation_width_factor,
                      int dilation_height_factor, PaddingType padding_type,
                      Dims<4>* output_dims, int* pad_width, int* pad_height) {
  const int input_width = ArraySize(input_dims, 1);
  const int input_height = ArraySize(input_dims, 2);
  const int batch = ArraySize(input_dims, 3);

  int dilated_filter_width = dilation_width_factor * (filter_width - 1) + 1;
  int dilated_filter_height = dilation_height_factor * (filter_height - 1) + 1;

  int output_height = 0;
  int output_width = 0;
  if (padding_type == PaddingType::kValid) {
    output_height = (input_height + stride - dilated_filter_height) / stride;
    output_width = (input_width + stride - dilated_filter_width) / stride;
  } else if (padding_type == PaddingType::kSame) {
    output_height = (input_height + stride - 1) / stride;
    output_width = (input_width + stride - 1) / stride;
  } else {
    return false;
  }

  if (output_width <= 0 || output_height <= 0) {
    return false;
  }

  *pad_height = std::max(
      0, ((output_height - 1) * stride + dilated_filter_height - input_height) /
             2);
  *pad_width = std::max(
      0,
      ((output_width - 1) * stride + dilated_filter_width - input_width) / 2);

  *output_dims =
      MakeDimsForInference(output_depth, output_width, output_height, batch);
  return true;
}

std::mt19937& RandomEngine() {
  static std::mt19937 engine;
  return engine;
}

int UniformRandomInt(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(RandomEngine());
}

float UniformRandomFloat(float min, float max) {
  std::uniform_real_distribution<float> dist(min, max);
  return dist(RandomEngine());
}

int ExponentialRandomPositiveInt(float percentile, int percentile_val,
                                 int max_val) {
  const float lambda =
      -std::log(1.f - percentile) / static_cast<float>(percentile_val);
  std::exponential_distribution<float> dist(lambda);
  float val;
  do {
    val = dist(RandomEngine());
  } while (!val || !std::isfinite(val) || val > max_val);
  return static_cast<int>(std::ceil(val));
}

float ExponentialRandomPositiveFloat(float percentile, float percentile_val,
                                     float max_val) {
  const float lambda =
      -std::log(1.f - percentile) / static_cast<float>(percentile_val);
  std::exponential_distribution<float> dist(lambda);
  float val;
  do {
    val = dist(RandomEngine());
  } while (!std::isfinite(val) || val > max_val);
  return val;
}

void FillRandom(std::vector<float>* vec, float min, float max) {
  std::uniform_real_distribution<float> dist(min, max);
  auto gen = std::bind(dist, RandomEngine());
  std::generate(std::begin(*vec), std::end(*vec), gen);
}

}  // namespace tflite
