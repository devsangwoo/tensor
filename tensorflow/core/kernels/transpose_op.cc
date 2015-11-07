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
// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/transpose_op.h"
<<<<<<< HEAD

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// inv = InvertPermutationOp(T<int32/int64> p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32 or int64.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.

template <typename T>
=======
#include "tensorflow/core/kernels/transpose_op_functor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// inv = InvertPermutationOp(T<int32> p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
class InvertPermutationOp : public OpKernel {
 public:
  explicit InvertPermutationOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input.shape()),
        errors::InvalidArgument("invert_permutation expects a 1D vector."));
<<<<<<< HEAD
    auto Tin = input.vec<T>();
    OP_REQUIRES(context,
                FastBoundsCheck(Tin.size(), std::numeric_limits<int32>::max()),
                errors::InvalidArgument("permutation of nonnegative int32s "
                                        "must have <= int32 max elements"));
    const T N = static_cast<T>(Tin.size());  // Safe: bounds-checked above.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    auto Tout = output->vec<T>();
    std::fill_n(Tout.data(), N, -1);
    for (int i = 0; i < N; ++i) {
      const T d = internal::SubtleMustCopy(Tin(i));
      OP_REQUIRES(context, FastBoundsCheck(d, N),
=======
    auto Tin = input.vec<int32>();
    const int N = Tin.size();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    auto Tout = output->vec<int32>();
    std::fill_n(Tout.data(), N, -1);
    for (int i = 0; i < N; ++i) {
      const int32 d = Tin(i);
      OP_REQUIRES(context, 0 <= d && d < N,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                  errors::InvalidArgument(d, " is not between 0 and ", N));
      OP_REQUIRES(context, Tout(d) == -1,
                  errors::InvalidArgument(d, " is duplicated in the input."));
      Tout(d) = i;
    }
  }
};

<<<<<<< HEAD
REGISTER_KERNEL_BUILDER(
    Name("InvertPermutation").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("InvertPermutation").Device(DEVICE_CPU).TypeConstraint<int64>("T"),
    InvertPermutationOp<int64>);

REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int64>);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int64>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int64>);
#endif  // TENSORFLOW_USE_SYCL

namespace {
template <typename Tperm>
Status PermutationHelper(const Tensor& perm, const int dims,
                         std::vector<int32>* permutation) {
  auto Vperm = perm.vec<Tperm>();
  if (dims != Vperm.size()) {
    return errors::InvalidArgument("transpose expects a vector of size ", dims,
                                   ". But input(1) is a vector of size ",
                                   Vperm.size());
  }
  // using volatile instead of SubtleMustCopy here so that the
  // asynchrony boundary is permutation.
  const volatile Tperm* perm_begin =
      reinterpret_cast<const volatile Tperm*>(Vperm.data());
  *permutation = std::vector<int32>(perm_begin, perm_begin + dims);

  return Status::OK();
}
}  // namespace
=======
REGISTER_KERNEL_BUILDER(Name("InvertPermutation").Device(DEVICE_CPU),
                        InvertPermutationOp);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// output = TransposeOp(T<any> input, T<int32> perm) takes a tensor
// of type T and rank N, and a permutation of 0, 1, ..., N-1. It
// shuffles the dimensions of the input tensor according to permutation.
//
// Specifically, the returned tensor output meets the following condition:
// 1) output.dims() == input.dims();
// 2) output.dim_size(i) == input.dim_size(perm[i]);
// 3) output.tensor<T, N>(i_0, i_1, ..., i_N-1) ==
//      input.tensor<T, N>(j_0, j_1, ..., j_N-1),
//    where i_s == j_{perm[s]}
//
// REQUIRES: perm is a vector of int32.
// REQUIRES: input.dims() == perm.size().
// REQUIRES: perm is a permutation.

<<<<<<< HEAD
void TransposeOp::Compute(OpKernelContext* ctx) {
  const Tensor& input = ctx->input(0);
  const Tensor& perm = ctx->input(1);
  // Preliminary validation of sizes.
  OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm.shape()),
              errors::InvalidArgument("perm must be a vector, not ",
                                      perm.shape().DebugString()));

  // Although Tperm may be an int64 type, an int32 is sufficient to hold
  // dimension range values, so the narrowing here should be safe.
  std::vector<int32> permutation;
  const int dims = input.dims();
  if (perm.dtype() == DT_INT32) {
    OP_REQUIRES_OK(ctx, PermutationHelper<int32>(perm, dims, &permutation));
  } else {
    OP_REQUIRES_OK(ctx, PermutationHelper<int64>(perm, dims, &permutation));
  }
=======
template <typename Device, typename T>
TransposeOp<Device, T>::TransposeOp(OpKernelConstruction* context)
    : OpKernel(context) {}

template <typename Device, typename T>
void TransposeOp<Device, T>::Compute(OpKernelContext* context) {
  const Tensor& input = context->input(0);
  const Tensor& perm = context->input(1);
  // Preliminary validation of sizes.
  OP_REQUIRES(context, TensorShapeUtils::IsVector(perm.shape()),
              errors::InvalidArgument("perm must be a vector, not ",
                                      perm.shape().DebugString()));
  auto Vperm = perm.vec<int32>();
  const int dims = input.dims();
  static const int kMinDims = 1;
  static const int kMaxDims = 8;
  OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
              errors::Unimplemented("Transposing a tensor of rank ", dims,
                                    " is not implemented."));
  OP_REQUIRES(context, dims == Vperm.size(),
              errors::InvalidArgument(
                  "transpose expects a vector of size ", input.dims(),
                  ". But input(1) is a vector of size ", Vperm.size()));
  gtl::ArraySlice<int32> permutation(
      reinterpret_cast<const int32*>(Vperm.data()), dims);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  TensorShape shape;

  // Check whether permutation is a permutation of integers of [0 .. dims).
  gtl::InlinedVector<bool, 8> bits(dims);
<<<<<<< HEAD
  bool is_identity = true;
  for (int i = 0; i < dims; ++i) {
    const int32 d = permutation[i];
    OP_REQUIRES(
        ctx, 0 <= d && d < dims,
        errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
    bits[d] = true;
    const auto dim_size = input.dim_size(d);
    shape.AddDim(dim_size);
    if (d != i) {
      is_identity = false;
    }
  }
  for (int i = 0; i < dims; ++i) {
    OP_REQUIRES(ctx, bits[i],
                errors::InvalidArgument(i, " is missing from {",
                                        absl::StrJoin(permutation, ","), "}."));
  }

  // 0-D, 1-D, and identity transposes do nothing.
  if (!IsConjugate() && (dims <= 1 || is_identity)) {
    ctx->set_output(0, input);
    return;
  } else if (!IsConjugate() && internal::NonSingletonDimensionsAlign(
                                   input.shape(), permutation)) {
    Tensor output;
    OP_REQUIRES(ctx, output.CopyFrom(input, shape),
                errors::Unknown("Error reshaping Tensor."));
    ctx->set_output(0, output);
    return;
  }

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  if (shape.num_elements() > 0) {
    OP_REQUIRES_OK(ctx, DoTranspose(ctx, input, permutation, output));
  }
}

Status TransposeCpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                   gtl::ArraySlice<int32> perm, Tensor* out) {
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
}

Status ConjugateTransposeCpuOp::DoTranspose(OpKernelContext* ctx,
                                            const Tensor& in,
                                            gtl::ArraySlice<int32> perm,
                                            Tensor* out) {
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<CPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeCpuOp);            \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeCpuOp);

TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
Status TransposeGpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                   gtl::ArraySlice<int32> perm, Tensor* out) {
  typedef Eigen::GpuDevice GPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<GPUDevice>(), in, perm,
                                   out);
}
Status ConjugateTransposeGpuOp::DoTranspose(OpKernelContext* ctx,
                                            const Tensor& in,
                                            gtl::ArraySlice<int32> perm,
                                            Tensor* out) {
  typedef Eigen::GpuDevice GPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<GPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeGpuOp);            \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeGpuOp);
TF_CALL_POD_TYPES(REGISTER);
#undef REGISTER
#endif

#ifdef TENSORFLOW_USE_SYCL
Status TransposeSyclOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                    gtl::ArraySlice<int32> perm, Tensor* out) {
  typedef Eigen::SyclDevice SYCLDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<SYCLDevice>(), in, perm,
                                   out);
}
Status ConjugateTransposeSyclOp::DoTranspose(OpKernelContext* ctx,
                                             const Tensor& in,
                                             gtl::ArraySlice<int32> perm,
                                             Tensor* out) {
  typedef Eigen::SyclDevice SYCLDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<SYCLDevice>(), in,
                                            perm, out);
}
#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_SYCL)    \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeSyclOp);           \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_SYCL)    \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeSyclOp);
TF_CALL_POD_TYPES(REGISTER);
#undef REGISTER
#endif
=======
  for (const int32 d : permutation) {
    OP_REQUIRES(
        context, 0 <= d && d < dims,
        errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
    bits[d] = true;
    shape.AddDim(input.dim_size(d));
  }
  for (int i = 0; i < dims; ++i) {
    OP_REQUIRES(context, bits[i], errors::InvalidArgument(
                                      i, " is missing from {",
                                      str_util::Join(permutation, ","), "}."));
  }

  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
  switch (dims) {
#define EXPAND_DIM(N)                                             \
  case N: {                                                       \
    functor::TransposeFunctor<Device, T, N> func;                 \
    func(context->eigen_device<Device>(), output->tensor<T, N>(), \
         input.tensor<T, N>(), permutation.data());               \
    break;                                                        \
  }
    EXPAND_DIM(1);
    EXPAND_DIM(2);
    EXPAND_DIM(3);
    EXPAND_DIM(4);
    EXPAND_DIM(5);
    EXPAND_DIM(6);
    EXPAND_DIM(7);
    EXPAND_DIM(8);
    default:
      LOG(FATAL) << "Unexpected dims: " << dims;
  }
#undef EXPAND_CASE
}

namespace functor {

template <typename Device, typename T, int NDIMS>
void TransposeMaybeInline(const Device& d,
                          typename TTypes<T, NDIMS>::Tensor out,
                          typename TTypes<T, NDIMS>::ConstTensor in,
                          const int* perm) {
  // perm[] is a permutation of 0, 1, ..., NDIMS-1. perm[] is on CPU.
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
  if (out.size() * sizeof(T) < 131072) {  // Small transpose on a CPU: do inline
    out = in.shuffle(p);
  } else {
    out.device(d) = in.shuffle(p);
  }
}

template <typename T, int NDIMS>
struct TransposeFunctor<CPUDevice, T, NDIMS> {
  void operator()(const CPUDevice& d, typename TTypes<T, NDIMS>::Tensor out,
                  typename TTypes<T, NDIMS>::ConstTensor in, const int* perm) {
    TransposeMaybeInline<CPUDevice, T, NDIMS>(d, out, in, perm);
  }
};

}  // namespace functor

#define REGISTER(D, T)                                \
  template class TransposeOp<D##Device, T>;           \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_##D)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeOp<D##Device, T>)
REGISTER(CPU, float);
REGISTER(CPU, double);
REGISTER(CPU, complex64);
REGISTER(CPU, uint8);
REGISTER(CPU, int8);
REGISTER(CPU, int16);
REGISTER(CPU, int32);
REGISTER(CPU, int64);
REGISTER(CPU, string);
#if GOOGLE_CUDA
REGISTER(GPU, uint8);
REGISTER(GPU, int8);
REGISTER(GPU, int16);
REGISTER(GPU, int32);
REGISTER(GPU, int64);
REGISTER(GPU, float);
REGISTER(GPU, double);
#endif
#undef REGISTER
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
