/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {
template <typename T, size_t NumDims, size_t DoubleNumDims>
class DiagonalGenerator {
 public:
  explicit DiagonalGenerator(const Tensor& diagonal) : diagonal_(diagonal) {
    static_assert(DoubleNumDims == 2 * NumDims,
                  "The second size must be the double of the first size.");
    CHECK_EQ(diagonal.dims(), NumDims);
  }
  T operator()(
      const Eigen::array<Eigen::DenseIndex, DoubleNumDims>& coordinates) const {
    Eigen::array<Eigen::DenseIndex, NumDims> index;
    for (size_t i = 0; i < NumDims; ++i) {
      if (coordinates[i] != coordinates[NumDims + i]) {
        return T(0);
      }
      index[i] = coordinates[i];
    }
    return diagonal_.tensor<T, NumDims>()(index);
  }

 private:
  Tensor diagonal_;
};
}  // namespace

// Generate the diagonal tensor with the diagonal set to the input tensor.
// It only allows up to rank 3 input tensor, so the output tensor is up to
// rank 6.
template <typename T>
class DiagOp : public OpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& diagonal = context->input(0);
    const int num_dims = diagonal.dims();
    OP_REQUIRES(context, 1 <= num_dims,
                errors::InvalidArgument(
                    "The rank of the diagonal should be between 1 and 3."));
    OP_REQUIRES(context, 3 >= num_dims,
                errors::InvalidArgument(
                    "The rank of the diagonal  should be between 1 and 3."));
    TensorShape out_shape;
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));
    switch (num_dims) {
      case 1:
        output_tensor->tensor<T, 2>() = output_tensor->tensor<T, 2>().generate(
            DiagonalGenerator<T, 1, 2>(diagonal));
        break;
      case 2:
        output_tensor->tensor<T, 4>() = output_tensor->tensor<T, 4>().generate(
            DiagonalGenerator<T, 2, 4>(diagonal));
        break;
      case 3:
        output_tensor->tensor<T, 6>() = output_tensor->tensor<T, 6>().generate(
            DiagonalGenerator<T, 3, 6>(diagonal));
        break;
      default:
        context->SetStatus(errors::Unimplemented(
            "Diagonal of rank ", num_dims, " tensor is not supported yet."));
        return;
    }
  }
};

#define REGISTER_DIAGOP(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("Diag").Device(DEVICE_CPU).TypeConstraint<T>("T"), DiagOp<T>)

REGISTER_DIAGOP(double);
REGISTER_DIAGOP(float);
REGISTER_DIAGOP(int32);
REGISTER_DIAGOP(int64);

#undef REGISTER_DIAGOP
}  // namespace tensorflow
