<<<<<<< HEAD
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
=======
#include "tensorflow/core/framework/allocator.h"
#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
<<<<<<< HEAD
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class AdjustContrastOpTest : public OpsTestBase {};

TEST_F(AdjustContrastOpTest, Simple_1113) {
  TF_EXPECT_OK(NodeDefBuilder("adjust_contrast_op", "AdjustContrastv2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 1, 3}), {-1, 2, 3});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 3}));
  test::FillValues<float>(&expected, {-1, 2, 3});
=======
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

class AdjustContrastOpTest : public OpsTestBase {
 protected:
  void MakeOp() { RequireDefaultOps(); }
};

TEST_F(AdjustContrastOpTest, Simple_1113) {
  RequireDefaultOps();
  EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrast")
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(node_def()));
  EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 1, 3}), {-1, 2, 3});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  AddInputFromArray<float>(TensorShape({}), {0.0});
  AddInputFromArray<float>(TensorShape({}), {2.0});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 3}));
  test::FillValues<float>(&expected, {0, 2, 2});
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(AdjustContrastOpTest, Simple_1223) {
<<<<<<< HEAD
  TF_EXPECT_OK(NodeDefBuilder("adjust_contrast_op", "AdjustContrastv2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2, 2, 3}),
                           {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
  AddInputFromArray<float>(TensorShape({}), {0.2f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 3}));
  test::FillValues<float>(&expected, {2.2, 6.2, 10.2, 2.4, 6.4, 10.4, 2.6, 6.6,
                                      10.6, 2.8, 6.8, 10.8});
=======
  RequireDefaultOps();
  EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrast")
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(node_def()));
  EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2, 2, 3}),
                           {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
  AddInputFromArray<float>(TensorShape({}), {0.2});
  AddInputFromArray<float>(TensorShape({}), {0.0});
  AddInputFromArray<float>(TensorShape({}), {10.0});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 3}));
  test::FillValues<float>(
      &expected, {2.2, 6.2, 10, 2.4, 6.4, 10, 2.6, 6.6, 10, 2.8, 6.8, 10});
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(AdjustContrastOpTest, Big_99x99x3) {
<<<<<<< HEAD
  TF_EXPECT_OK(NodeDefBuilder("adjust_contrast_op", "AdjustContrastv2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  std::vector<float> values;
  values.reserve(99 * 99 * 3);
=======
  EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrast")
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(node_def()));
  EXPECT_OK(InitOp());

  std::vector<float> values;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  for (int i = 0; i < 99 * 99 * 3; ++i) {
    values.push_back(i % 255);
  }

  AddInputFromArray<float>(TensorShape({1, 99, 99, 3}), values);
<<<<<<< HEAD
  AddInputFromArray<float>(TensorShape({}), {0.2f});
  TF_ASSERT_OK(RunOpKernel());
=======
  AddInputFromArray<float>(TensorShape({}), {0.2});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255});
  ASSERT_OK(RunOpKernel());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

}  // namespace tensorflow
