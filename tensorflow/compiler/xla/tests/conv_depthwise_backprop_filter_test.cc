/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/despecializer.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

string GetFloatDataType(bool use_bfloat16) {
  return use_bfloat16 ? "bf16" : "f32";
}

struct DepthwiseConvolution2DSpec {
  int64 output_batch, window;
  std::vector<int64> activation_dims;
  std::vector<int64> activation_layout;
  std::vector<int64> kernel_dims;
  std::vector<int64> kernel_layout;
  std::vector<int64> output_dims;
  std::vector<int64> output_layout;
};

class DepthwiseConvolution2DTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ::testing::tuple<DepthwiseConvolution2DSpec, bool>> {};

static std::vector<DepthwiseConvolution2DSpec> GetConv2DTestCases() {
  std::vector<DepthwiseConvolution2DSpec> config_set;
  std::vector<std::vector<int64>> config_options = {
      {16, 5, 5, 2}, {64, 4, 4, 16}, {2, 5, 5, 256}};

  for (auto option : config_options) {
    int64 feature = option[3];
    int64 activation_size = option[1];
    int64 kernel_size = option[2];
    int64 batch = option[0];

    std::vector<int64> kernel_layout = {3, 2, 1, 0};
    DepthwiseConvolution2DSpec config;
    config.output_batch = feature;
    config.window = kernel_size;

    config.activation_dims = {batch, activation_size, activation_size, feature};
    config.activation_layout = {0, 3, 2, 1};

    config.kernel_dims = {batch, kernel_size, kernel_size, feature};
    config.kernel_layout = {0, 2, 3, 1};

    config.output_dims = {3, 3, feature, 1};

    // Try this layout for all kernel shapes.
    config.output_layout = {3, 2, 0, 1};
    config_set.push_back(config);
  }

  return config_set;
}

string DepthwiseConvolution2DTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<DepthwiseConvolution2DSpec, bool>>& data) {
  const auto& spec = ::testing::get<0>(data.param);
  const string data_type = GetFloatDataType(::testing::get<1>(data.param));
  string str = absl::StrCat(
      "activation_dims_", absl::StrJoin(spec.activation_dims, "x"),
      "_activation_layout_", absl::StrJoin(spec.activation_layout, "_"),
      "_kernel_dims_", absl::StrJoin(spec.kernel_dims, "x"), "_kernel_layout_",
      absl::StrJoin(spec.kernel_layout, "_"), "_output_dims_",
      absl::StrJoin(spec.output_dims, "x"), "_output_layout_",
      absl::StrJoin(spec.output_layout, "_"), data_type);

  // Test names are not allowed to contain the '-' character.
  absl::c_replace(str, '-', 'n');
  return str;
}

string BuildHloTextDepthwiseConvolution2D(
    const DepthwiseConvolution2DSpec& spec, bool use_bfloat16) {
  const string data_type = GetFloatDataType(use_bfloat16);
  return absl::StrFormat(
      R"(
    HloModule TensorFlowDepthwiseConv

    ENTRY main {
      activation = %s[%s]{%s} parameter(0)
      kernel = %s[%s]{%s} parameter(1)
      ROOT conv = %s[%s]{%s} convolution(%s[%s]{%s} activation, %s[%s]{%s} kernel),
          window={size=%dx%d pad=1_1x1_1}, dim_labels=f01b_i01o->01fb,
          batch_group_count=%d
    }
    )",
      data_type, absl::StrJoin(spec.activation_dims, ","),
      absl::StrJoin(spec.activation_layout, ","), data_type,
      absl::StrJoin(spec.kernel_dims, ","),
      absl::StrJoin(spec.kernel_layout, ","), data_type,
      absl::StrJoin(spec.output_dims, ","),
      absl::StrJoin(spec.output_layout, ","), data_type,
      absl::StrJoin(spec.activation_dims, ","),
      absl::StrJoin(spec.activation_layout, ","), data_type,
      absl::StrJoin(spec.kernel_dims, ","),
      absl::StrJoin(spec.kernel_layout, ","), spec.window, spec.window,
      spec.output_batch);
}

XLA_TEST_P(DepthwiseConvolution2DTest, DoIt) {
  const DepthwiseConvolution2DSpec& spec = ::testing::get<0>(GetParam());
  bool use_bfloat16 = ::testing::get<1>(GetParam());
  const string hlo_text =
      BuildHloTextDepthwiseConvolution2D(spec, use_bfloat16);

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0.01, 0.01},
                            [](HloModule* module) -> Status {
                              BFloat16MixedPrecisionRemoval remover;
                              TF_RETURN_IF_ERROR(remover.Run(module).status());
                              Despecializer despecializer;
                              return despecializer.Run(module).status();
                            }));
}

INSTANTIATE_TEST_CASE_P(
    DepthwiseConvolution2DTestWithRandomIndices, DepthwiseConvolution2DTest,
    ::testing::Combine(::testing::ValuesIn(GetConv2DTestCases()),
                       ::testing::Bool()),
    DepthwiseConvolution2DTestDataToString);

}  // namespace
}  // namespace xla
