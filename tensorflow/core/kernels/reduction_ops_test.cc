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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
=======
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include <gtest/gtest.h>
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

// Creates a Graph which "reduce"s a 3D float tensor of "num" elements
// into a scalar.
<<<<<<< HEAD
template <typename T>
static Graph* ToScalar(const string& reduce, int num_x, int num_y) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DataTypeToEnum<T>::value, TensorShape({num_x, num_y}));
  data.flat<T>().setRandom();
  Tensor axes(DT_INT32, TensorShape({2}));
  axes.flat<int32>()(0) = 0;
  axes.flat<int32>()(1) = 1;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ColReduce(const string& reduce, int num_x, int num_y) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({num_x, num_y}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = 0;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* RowReduce(const string& reduce, int num_x, int num_y) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({num_x, num_y}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = 1;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ThreeDYReduce(const string& reduce, int num_y, int num_z) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({4, num_y, num_z}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = 1;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ThreeDXZReduce(const string& reduce, int num_y, int num_z) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({4, num_y, num_z}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({2}));
  axes.flat<int32>()(0) = 0;
  axes.flat<int32>()(1) = 2;
=======
static Graph* ToScalar(const string& reduce, int num) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({64, 64, num / (64 * 64)}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({3}));
  axes.flat<int32>()(0) = 0;
  axes.flat<int32>()(1) = 1;
  axes.flat<int32>()(2) = 2;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

// Creates a bench which reduces a 3D tensor with total "num" floats
// into a scalar on a "device". Runs the bench for "iters" times.
<<<<<<< HEAD
template <typename T>
static void ReduceToScalar(int iters, const string& device,
                           const string& reduce, int num_x, int num_y) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num_x * num_y);
  testing::BytesProcessed(static_cast<int64>(iters) * num_x * num_y *
                          sizeof(T));
  test::Benchmark(device, ToScalar<T>(reduce, num_x, num_y)).Run(iters);
}

static void DoRowReduce(int iters, const string& device, const string& reduce,
                        int num_x, int num_y) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num_x * num_y);
  testing::BytesProcessed(static_cast<int64>(iters) * num_x * num_y *
                          sizeof(float));
  test::Benchmark(device, RowReduce(reduce, num_x, num_y)).Run(iters);
}

static void DoColReduce(int iters, const string& device, const string& reduce,
                        int num_x, int num_y) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num_x * num_y);
  testing::BytesProcessed(static_cast<int64>(iters) * num_x * num_y *
                          sizeof(float));
  test::Benchmark(device, ColReduce(reduce, num_x, num_y)).Run(iters);
}

static void Do3DYReduce(int iters, const string& device, const string& reduce,
                        int num_x, int num_y) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num_x * num_y);
  testing::BytesProcessed(static_cast<int64>(iters) * num_x * num_y *
                          sizeof(float));
  test::Benchmark(device, ThreeDYReduce(reduce, num_x, num_y)).Run(iters);
}

static void Do3DXZReduce(int iters, const string& device, const string& reduce,
                         int num_x, int num_y) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num_x * num_y);
  testing::BytesProcessed(static_cast<int64>(iters) * num_x * num_y *
                          sizeof(float));
  test::Benchmark(device, ThreeDXZReduce(reduce, num_x, num_y)).Run(iters);
}

static void BM_Sum2DToScalarGPU(int iters, int num_x, int num_y) {
  ReduceToScalar<float>(iters, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DToScalarGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DToScalarGPUComplex(int iters, int num_x, int num_y) {
  ReduceToScalar<std::complex<float>>(iters, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DToScalarGPUComplex)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DToScalarGPUHalf(int iters, int num_x, int num_y) {
  ReduceToScalar<Eigen::half>(iters, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DToScalarGPUHalf)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DRowReduceGPU(int iters, int num_x, int num_y) {
  DoRowReduce(iters, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DRowReduceGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DColumnReduceGPU(int iters, int num_x, int num_y) {
  DoColReduce(iters, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DColumnReduceGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum3DYReduceGPU(int iters, int num_x, int num_y) {
  Do3DYReduce(iters, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum3DYReduceGPU)->RangePair(64, 4096, 64, 4096);

static void BM_Sum3DXZReduceGPU(int iters, int num_x, int num_y) {
  Do3DXZReduce(iters, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum3DXZReduceGPU)->RangePair(64, 4096, 64, 4096);

static void BM_Mean2DToScalarGPU(int iters, int num_x, int num_y) {
  ReduceToScalar<float>(iters, "gpu", "Mean", num_x, num_y);
}
BENCHMARK(BM_Mean2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_EuclideanNorm2DToScalarGPU(int iters, int num_x, int num_y) {
  ReduceToScalar<float>(iters, "gpu", "EuclideanNorm", num_x, num_y);
}
BENCHMARK(BM_EuclideanNorm2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_Max2DToScalarGPU(int iters, int num_x, int num_y) {
  ReduceToScalar<float>(iters, "gpu", "Max", num_x, num_y);
}
BENCHMARK(BM_Max2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_Min2DToScalarGPU(int iters, int num_x, int num_y) {
  ReduceToScalar<float>(iters, "gpu", "Min", num_x, num_y);
}
BENCHMARK(BM_Min2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_Min2DToScalarGPUHalf(int iters, int num_x, int num_y) {
  ReduceToScalar<Eigen::half>(iters, "gpu", "Min", num_x, num_y);
}
BENCHMARK(BM_Min2DToScalarGPUHalf)->RangePair(2048, 8192, 2048, 8192);

static void BM_Bool2DToScalarGPU(int iters, int num_x, int num_y) {
  ReduceToScalar<bool>(iters, "gpu", "All", num_x, num_y);
}
BENCHMARK(BM_Bool2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);
=======
static void ReduceToScalar(int iters, const string& device,
                           const string& reduce, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num * sizeof(float));
  test::Benchmark(device, ToScalar(reduce, num)).Run(iters);
}

static void BM_Sum3DToScalarCPU(int iters, int num) {
  ReduceToScalar(iters, "cpu", "Sum", num);
}
BENCHMARK(BM_Sum3DToScalarCPU)->Range(1 << 13, 1 << 20);

static void BM_Max3DToScalarCPU(int iters, int num) {
  ReduceToScalar(iters, "cpu", "Max", num);
}
BENCHMARK(BM_Max3DToScalarCPU)->Range(1 << 13, 1 << 20);

static void BM_Prod3DToScalarCPU(int iters, int num) {
  ReduceToScalar(iters, "cpu", "Prod", num);
}
BENCHMARK(BM_Prod3DToScalarCPU)->Range(1 << 13, 1 << 20);

static void BM_Mean3DToScalarCPU(int iters, int num) {
  ReduceToScalar(iters, "cpu", "Mean", num);
}
BENCHMARK(BM_Mean3DToScalarCPU)->Range(1 << 13, 1 << 20);

static void BM_Sum3DToScalarGPU(int iters, int num) {
  ReduceToScalar(iters, "gpu", "Sum", num);
}
BENCHMARK(BM_Sum3DToScalarGPU)->Range(1 << 13, 1 << 20);

static void BM_Max3DToScalarGPU(int iters, int num) {
  ReduceToScalar(iters, "gpu", "Max", num);
}
BENCHMARK(BM_Max3DToScalarGPU)->Range(1 << 13, 1 << 20);

static void BM_Prod3DToScalarGPU(int iters, int num) {
  ReduceToScalar(iters, "gpu", "Prod", num);
}
BENCHMARK(BM_Prod3DToScalarGPU)->Range(1 << 13, 1 << 20);

// Once Mean is available on GPU, enable this.
// static void BM_Mean3DToScalarGPU(int iters, int num) {
//   ReduceToScalar(iters, "gpu", "Mean", num);
// }
// BENCHMARK(BM_Mean3DToScalarGPU)->Range(1 << 13, 1 << 20);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // end namespace tensorflow
