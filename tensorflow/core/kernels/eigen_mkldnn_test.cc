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

#include "tensorflow/core/kernels/eigen_mkldnn.h"
#include "tensorflow/core/platform/test.h"

namespace Eigen {
namespace internal {

namespace {
template <typename Index, int NumDims>
Eigen::array<Index, NumDims> RandomDims(int min_dim = 1, int max_dim = 20) {
  Eigen::array<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<int>(min_dim, max_dim);
  }
  return dims;
}
}  // namespace

using Scalar = float;
using Index = Eigen::Index;

TEST(EigenMkldnnTest, MkldnnPack) {
  // Packing with mkldnn_gemm_pack is the same as taking a slice of 2
  // dimensional Tensor.

  // Mkldnn pack and gemm are used only in Tensor contractions, and it's
  // guaranteed that Tensors will have ColMajor layout.
  static const int Options = ColMajor;

  using DataMapper = blas_data_mapper<Scalar, Index, ColMajor>;
  using MkldnnGemmPack = mkldnn_gemm_pack<Scalar, Index, DataMapper, ColMajor>;
  using Tensor2d = Tensor<Scalar, 2, Options, Index>;

  Eigen::array<Index, 2> dims = RandomDims<Index, 2>(1, 500);

  // Create a tensor initialized with random data.
  Tensor2d src(dims);
  src.setRandom();

  // Pick a random slice of src tensor.
  Eigen::array<Index, 2> slice_start = RandomDims<Index, 2>(0, 250);
  Eigen::array<Index, 2> slice_size = RandomDims<Index, 2>(100, 500);

  // Make sure that slice start + size do not overflow tensor dims.
  for (int i = 0; i < 2; ++i) {
    slice_start[i] = numext::mini(dims[i] - 1, slice_start[i]);
    slice_size[i] = numext::mini(slice_size[i], dims[i] - slice_start[i]);
  }

  // Prepare tensors for packing and slicing results.
  Tensor2d pack_dst(slice_size[0], slice_size[1]);
  Tensor2d slice_dst(slice_size[0], slice_size[1]);

  // Pack memory using mkldnn_gemm_pack.
  DataMapper data_mapper(src.data(), dims[0]);
  MkldnnGemmPack gemm_pack;
  gemm_pack(pack_dst.data(),
            data_mapper.getSubMapper(slice_start[0], slice_start[1]),
            slice_size[0], slice_size[1]);

  // Slice the source tensor.
  slice_dst = src.slice(slice_start, slice_size);

  // Verify that dst tensors are equal.
  EXPECT_EQ(pack_dst.dimensions().TotalSize(),
            slice_dst.dimensions().TotalSize());
  for (size_t i = 0; i < pack_dst.dimensions().TotalSize(); ++i) {
    Scalar packed = pack_dst.coeff(i);
    Scalar sliced = slice_dst.coeff(i);
    EXPECT_EQ(packed, sliced);
  }
}

TEST(EigenMkldnnTest, MkldnnGemm) {
  // Mkldnn pack and gemm are used only in Tensor contractions, and it's
  // guaranteed that Tensors will have ColMajor layout.
  static const int Options = ColMajor;

  using Tensor2d = Tensor<Scalar, 2, Options, Index>;

  int m = internal::random<int>(1, 100);
  int n = internal::random<int>(1, 100);
  int k = internal::random<int>(1, 100);

  Tensor2d lhs(m, k);
  lhs.setRandom();

  Tensor2d rhs(k, n);
  rhs.setRandom();

  // Compute matmul with mkldnn gemm kernel.
  using OutputMapper = blas_data_mapper<Scalar, Index, ColMajor>;
  using MkldnnGemmKernel =
      mkldnn_gemm_kernel<Scalar, Index, OutputMapper, ColMajor>;

  Tensor2d mkldnn_result(m, n);
  mkldnn_result.setZero();
  OutputMapper output_mapper(mkldnn_result.data(), m);

  MkldnnGemmKernel gemm_kernel;
  gemm_kernel(output_mapper, lhs.data(), rhs.data(), m, k, n, /*alpha=*/1.0);

  // Compute matmul with Eigen::Matrix.
  using Matrix = Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using MatrixMap = Map<Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor>>;

  MatrixMap lhs_mat(lhs.data(), m, k);
  MatrixMap rhs_mat(rhs.data(), k, n);

  Matrix matmul_result(m, n);
  matmul_result.setZero();
  matmul_result = lhs_mat * rhs_mat;

  // Verify that results are equal.
  for (Index i = 0; i < m * n; ++i) {
    Scalar gemm = mkldnn_result(i);
    Scalar matmul = matmul_result(i % m, i / m);

    Scalar delta = std::abs(gemm - matmul);

    // NOTE(rmlarsen): Compute proper forward error bound.
    Scalar sum = Scalar(0.0);
    for (int k1 = 0; k1 < k; ++k1) {
      sum += std::abs(lhs_mat(i % m, k1) * rhs_mat(k1, i / m));
    }
    Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
    Scalar upper_bound = Scalar(1.01) * epsilon * k * sum;

    EXPECT_LE(delta, upper_bound);
  }
}

}  // namespace internal
}  // namespace Eigen
