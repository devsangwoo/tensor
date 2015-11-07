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

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include <functional>
#include <memory>
#include <vector>

<<<<<<< HEAD
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/testlib.h"
=======
#include <gtest/gtest.h>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
<<<<<<< HEAD
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
=======
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/tensor.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {
namespace {

class GatherOpTest : public OpsTestBase {
 protected:
<<<<<<< HEAD
  void MakeOp(DataType data_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "GatherV2")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(index_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
=======
  void MakeOp(DataType index_type) {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("myop", "Gather")
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(index_type))
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
};

TEST_F(GatherOpTest, ScalarIndices) {
<<<<<<< HEAD
  MakeOp(DT_FLOAT, DT_INT32);
=======
  MakeOp(DT_INT32);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {3});
<<<<<<< HEAD
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());
=======
  ASSERT_OK(RunOpKernel());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected, {3});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

<<<<<<< HEAD
TEST_F(GatherOpTest, ScalarIndices_Complex) {
  MakeOp(DT_COMPLEX64, DT_INT32);

  // Feed and run
  AddInputFromArray<std::complex<float>>(
      TensorShape({5}), {std::complex<float>(0, 10), std::complex<float>(1, 11),
                         std::complex<float>(2, 12), std::complex<float>(3, 13),
                         std::complex<float>(4, 14)});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_COMPLEX64, TensorShape({}));
  test::FillValues<std::complex<float>>(&expected,
                                        {std::complex<float>(3, 13)});
  test::ExpectTensorEqual<std::complex<float>>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD32_Axis0) {
  MakeOp(DT_FLOAT, DT_INT32);
=======
TEST_F(GatherOpTest, Simple_TwoD32) {
  MakeOp(DT_INT32);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 0, 2});
<<<<<<< HEAD
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());
=======
  ASSERT_OK(RunOpKernel());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 12, 13, 14, 0, 1, 2, 6, 7, 8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

<<<<<<< HEAD
TEST_F(GatherOpTest, Simple_TwoD32_Axis1) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 1, 0, 2});
  AddInputFromArray<int32>(TensorShape({}), {1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 4}));
  test::FillValues<float>(&expected, {0, 1, 0, 2,  3, 4,  3,  5,  6,  7,
                                      6, 8, 9, 10, 9, 11, 12, 13, 12, 14});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, ZeroSize_TwoD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 0}), {});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 0, 2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 0}));
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD64) {
  MakeOp(DT_FLOAT, DT_INT64);
=======
TEST_F(GatherOpTest, Simple_TwoD64) {
  MakeOp(DT_INT64);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int64>(TensorShape({4}), {0, 4, 0, 2});
<<<<<<< HEAD
  AddInputFromArray<int64>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());
=======
  ASSERT_OK(RunOpKernel());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 12, 13, 14, 0, 1, 2, 6, 7, 8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, HighRank) {
<<<<<<< HEAD
  MakeOp(DT_FLOAT, DT_INT32);
=======
  MakeOp(DT_INT32);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Feed and run
  AddInputFromArray<float>(TensorShape({4}), {0, 1, 2, 3});
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 0, 2, 3, 0});
<<<<<<< HEAD
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());
=======
  ASSERT_OK(RunOpKernel());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected, {1, 2, 0, 2, 3, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Error_IndexOutOfRange) {
<<<<<<< HEAD
  MakeOp(DT_FLOAT, DT_INT32);
=======
  MakeOp(DT_INT32);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 99, 2});
<<<<<<< HEAD
  AddInputFromArray<int32>(TensorShape({}), {0});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "indices[2] = 99 is not in [0, 5)"))
      << s;
}

constexpr int kLookups = 2000;

template <typename Index>
static Graph* Gather(int dim) {
  Graph* g = new Graph(OpRegistry::Global());
  // Always use a 512MB buffer.
  const int kRows = ((512 << 20) / sizeof(float)) / dim;
  Tensor params(DT_FLOAT, TensorShape({kRows, dim}));
  params.flat<float>().setRandom();

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<Index> indices_vec;
  indices_vec.reserve(kLookups);
  for (int i = 0; i < kLookups; i++) {
    indices_vec.push_back(rnd.Uniform(kRows));
  }
  Tensor indices(DataTypeToEnum<Index>::value, TensorShape({kLookups}));
  for (int i = 0; i < indices_vec.size(); i++) {
    indices.flat<Index>()(i) = indices_vec[i];
  }

  Tensor axis(DataTypeToEnum<Index>::value, TensorShape({}));
  axis.scalar<Index>()() = 0;

  test::graph::Gather(g, test::graph::Constant(g, params),
                      test::graph::Constant(g, indices),
                      test::graph::HostConstant(g, axis));
  return g;
}

#define BM_GATHER(DEVICE, INDEX)                                  \
  static void BM_##DEVICE##_gather_##INDEX(int iters, int dim) {  \
    const int64 tot = static_cast<int64>(iters) * kLookups * dim; \
    testing::ItemsProcessed(tot);                                 \
    testing::BytesProcessed(tot * sizeof(float));                 \
    testing::UseRealTime();                                       \
    test::Benchmark(#DEVICE, Gather<INDEX>(dim)).Run(iters);      \
  }                                                               \
  BENCHMARK(BM_##DEVICE##_gather_##INDEX)                         \
      ->Arg(1)                                                    \
      ->Arg(10)                                                   \
      ->Arg(20)                                                   \
      ->Arg(64)                                                   \
      ->Arg(100)                                                  \
      ->Arg(200)                                                  \
      ->Arg(1000)

BM_GATHER(cpu, int32);
BM_GATHER(gpu, int32);
BM_GATHER(cpu, int64);
BM_GATHER(gpu, int64);
=======
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Index 99 at offset 2 in Tindices is out of range"))
      << s;
}

class GatherOpForBenchmark : public GatherOpTest {
 public:
  void TestBody() override {  // not used }
  }
  void PublicMakeOp(DataType index_type) { MakeOp(index_type); }
};

static const int kSorted = 0x8000;  // Mask for arg to specify sorting vs. not

template <typename Index>
void BM_Gather(int iters, int arg) {
  testing::StopTiming();

  bool sorted = ((arg & kSorted) != 0);
  int dim = arg & ~kSorted;

  GatherOpForBenchmark t;
  t.PublicMakeOp(DataTypeToEnum<Index>::v());
  // Use a 512 MB table, regardless of dim
  const int kRows = ((1 << 29) / sizeof(float)) / dim;
  std::vector<float> data(kRows * dim, 1.0f);
  t.AddInputFromArray<float>(TensorShape({kRows, dim}), data);
  const int kLookups = 2000;
  const int kBatches = 1000000 / kLookups;
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<std::vector<Index>> all_ids(kBatches);
  for (int i = 0; i < kBatches; ++i) {
    std::vector<Index>* ids = &all_ids[i];
    ids->resize(kLookups);
    for (int j = 0; j < kLookups; ++j) {
      (*ids)[j] = rnd.Uniform(kRows);
    }
    if (sorted) {
      sort(ids->begin(), ids->end());
    }
  }

  t.AddInput<Index>(TensorShape({kLookups}), [](int i) { return 0; });
  if (sorted) {
    testing::SetLabel("sorted by id");
  }
  testing::BytesProcessed(static_cast<int64>(iters) * kLookups * dim *
                          sizeof(float));
  testing::StartTiming();
  while (--iters > 0) {
    const std::vector<Index>& b = all_ids[iters % kBatches];
    TensorValue input = t.mutable_input(1);
    gtl::MutableArraySlice<Index> slice(&input->vec<Index>()(0),
                                        input->NumElements());
    for (int i = 0; i < kLookups; i++) {
      slice[i] = b[i];
    }
    Status s = t.RunOpKernel();
  }
}

static void BM_Gather32(int iters, int arg) { BM_Gather<int32>(iters, arg); }

static void BM_Gather64(int iters, int arg) { BM_Gather<int64>(iters, arg); }

BENCHMARK(BM_Gather32)
    ->Arg(10)
    ->Arg(10 | kSorted)
    ->Arg(20)
    ->Arg(40)
    ->Arg(63)
    ->Arg(63 | kSorted)
    ->Arg(64)
    ->Arg(64 | kSorted)
    ->Arg(65)
    ->Arg(65 | kSorted)
    ->Arg(100)
    ->Arg(100 | kSorted)
    ->Arg(127)
    ->Arg(127 | kSorted)
    ->Arg(128)
    ->Arg(128 | kSorted)
    ->Arg(129)
    ->Arg(129 | kSorted)
    ->Arg(1000)
    ->Arg(1000 | kSorted);

BENCHMARK(BM_Gather64)
    ->Arg(10)
    ->Arg(10 | kSorted)
    ->Arg(20)
    ->Arg(40)
    ->Arg(63)
    ->Arg(63 | kSorted)
    ->Arg(64)
    ->Arg(64 | kSorted)
    ->Arg(65)
    ->Arg(65 | kSorted)
    ->Arg(100)
    ->Arg(100 | kSorted)
    ->Arg(127)
    ->Arg(127 | kSorted)
    ->Arg(128)
    ->Arg(128 | kSorted)
    ->Arg(129)
    ->Arg(129 | kSorted)
    ->Arg(1000)
    ->Arg(1000 | kSorted);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace
}  // namespace tensorflow
