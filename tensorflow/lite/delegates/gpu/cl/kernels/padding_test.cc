/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/kernels/padding.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, PaddingAppendWidth) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = HWC(0, 0, 0);
  attr.appended = HWC(0, 1, 0);

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      Padding operation = CreatePadding(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 2, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps),
                            {0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 0.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, PaddingPrependWidth) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = HWC(0, 1, 0);
  attr.appended = HWC(0, 0, 0);

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      Padding operation = CreatePadding(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 2, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps),
                            {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 3.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, PaddingAppendHeight) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = HWC(0, 0, 0);
  attr.appended = HWC(1, 0, 0);

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      Padding operation = CreatePadding(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 3, 1, 2), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(eps), {0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, PaddingPrependHeight) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = HWC(1, 0, 0);
  attr.appended = HWC(0, 0, 0);

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      Padding operation = CreatePadding(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 3, 1, 2), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(eps), {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, PaddingAppendChannels) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = HWC(0, 0, 0);
  attr.appended = HWC(0, 0, 1);

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      Padding operation = CreatePadding(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 1, 3), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(eps), {0.0f, 1.0f, 0.0f, 2.0f, 3.0f, 0.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, PaddingPrependChannels) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = HWC(0, 0, 1);
  attr.appended = HWC(0, 0, 0);

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      Padding operation = CreatePadding(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 1, 3), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(eps), {0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 3.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, PaddingComplex) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = HWC(0, 1, 1);
  attr.appended = HWC(1, 1, 0);

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      Padding operation = CreatePadding(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 3, 3, 3), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(eps),
                    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
