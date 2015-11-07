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

// See docs in ../ops/array_ops.cc.

#include "tensorflow/core/kernels/shape_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

// Shape ----------------------------------------
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64>("out_type"),
                        ShapeOp<int64>);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int32>);                       \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int64>);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);
TF_CALL_bool(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeOp<int64>);
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int32>);                       \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int64>);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
TF_CALL_variant(REGISTER_GPU_KERNEL);
=======
// See docs in ../ops/array_ops.cc.

#include <unordered_set>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

class ShapeOp : public OpKernel {
 public:
  explicit ShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    const int rank = inp.dims();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rank}), &out));
    auto vec = out->vec<int32>();
    for (int i = 0; i < rank; ++i) vec(i) = inp.dim_size(i);
  }

  bool IsExpensive() override { return false; }
};
REGISTER_KERNEL_BUILDER(Name("Shape").Device(DEVICE_CPU).HostMemory("output"),
                        ShapeOp);

#define REGISTER_GPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Shape")                   \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ShapeOp)
TF_CALL_REAL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
<<<<<<< HEAD
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeOp<int64>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// ShapeN ---------------------------------------
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64>("out_type"),
                        ShapeNOp<int64>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int32>);                      \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int64>)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeNOp<int64>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int32>);                      \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int64>)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);
TF_CALL_bool(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeNOp<int64>);
#endif  // TENSORFLOW_USE_SYCL

// Rank ------------------------------------------
REGISTER_KERNEL_BUILDER(Name("Rank").Device(DEVICE_CPU).HostMemory("output"),
                        RankOp);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                       \
  REGISTER_KERNEL_BUILDER(Name("Rank")                   \
                              .Device(DEVICE_SYCL)       \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          RankOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
=======
                            .TypeConstraint<int32>("T"),
                        ShapeOp);

class RankOp : public OpKernel {
 public:
  explicit RankOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    const int rank = inp.dims();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<int32>()() = rank;
  }

  bool IsExpensive() override { return false; }
};
REGISTER_KERNEL_BUILDER(Name("Rank").Device(DEVICE_CPU).HostMemory("output"),
                        RankOp);

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#define REGISTER_GPU_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Rank")                   \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          RankOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
<<<<<<< HEAD
TF_CALL_variant(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32 and bool.
=======
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

<<<<<<< HEAD
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Size ------------------------------------------
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64>("out_type"),
                        SizeOp<int64>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int64>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
TF_CALL_variant(REGISTER_GPU_KERNEL);
=======
class SizeOp : public OpKernel {
 public:
  explicit SizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    const int64 size = inp.NumElements();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    // TODO(josh11b): switch output to int64?
    out->scalar<int32>()() = size;
  }

  bool IsExpensive() override { return false; }
};
REGISTER_KERNEL_BUILDER(Name("Size").Device(DEVICE_CPU).HostMemory("output"),
                        SizeOp);

#define REGISTER_GPU_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Size")                   \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          SizeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
<<<<<<< HEAD
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int64>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_SYCL)               \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_SYCL)               \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int64>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);
TF_CALL_bool(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int64>);
#endif  // TENSORFLOW_USE_SYCL

// ExpandDims ------------------------------------
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_CPU)
                            .HostMemory("dim")
                            .TypeConstraint<int32>("Tdim"),
                        ExpandDimsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_CPU)
                            .HostMemory("dim")
                            .TypeConstraint<int64>("Tdim"),
                        ExpandDimsOp<int64>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                            \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                 \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int32>("Tdim") \
                              .HostMemory("dim"),            \
                          ExpandDimsOp<int32>);              \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                 \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int64>("Tdim") \
                              .HostMemory("dim"),            \
                          ExpandDimsOp<int64>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int64>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                           \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                 \
                              .Device(DEVICE_SYCL)           \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int32>("Tdim") \
                              .HostMemory("dim"),            \
                          ExpandDimsOp<int32>);              \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                 \
                              .Device(DEVICE_SYCL)           \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int64>("Tdim") \
                              .HostMemory("dim"),            \
                          ExpandDimsOp<int64>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);
TF_CALL_bool(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int64>);
#endif  // TENSORFLOW_USE_SYCL

// Squeeze ---------------------------------------
REGISTER_KERNEL_BUILDER(Name("Squeeze").Device(DEVICE_CPU), SqueezeOp);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Squeeze").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SqueezeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Squeeze")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SqueezeOp);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Squeeze").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      SqueezeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);
TF_CALL_bool(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

REGISTER_KERNEL_BUILDER(Name("Squeeze")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SqueezeOp);
#endif  // TENSORFLOW_USE_SYCL

class EnsureShapeOp : public OpKernel {
 public:
  explicit EnsureShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &expected_shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, shape_op_helpers::GetShape(ctx, 0, &shape));

    if (!expected_shape_.IsCompatibleWith(shape)) {
      ctx->SetStatus(errors::InvalidArgument(
          "Shape of tensor ", this->def().input(0), " ", shape.DebugString(),
          " is not compatible with expected shape ",
          expected_shape_.DebugString(), "."));
    }

    // If shape matches, outputs the tensor.
    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      ctx->set_output(0, ctx->input(0));
    }
  }

  bool IsExpensive() override { return false; }

 private:
  PartialTensorShape expected_shape_;
};

// NOTE(rachelim): The kernel registrations for EnsureShapeOp are identical to
// those of the identity op, since the ops have the same device type
// constraints.
REGISTER_KERNEL_BUILDER(Name("EnsureShape").Device(DEVICE_CPU), EnsureShapeOp);

#if TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("EnsureShape").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      EnsureShapeOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#undef REGISTER_SYCL_KERNEL

#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("EnsureShape")             \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("input")        \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnsureShapeOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(bool);

#undef REGISTER_SYCL_HOST_KERNEL

#endif  // TENSORFLOW_USE_SYCL

#define REGISTER_GPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("EnsureShape").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      EnsureShapeOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
REGISTER_GPU_KERNEL(Variant);

#undef REGISTER_GPU_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("EnsureShape")             \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("input")        \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnsureShapeOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(bool);
REGISTER_GPU_HOST_KERNEL(tstring);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL

#endif
=======
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp);

class ExpandDimsOp : public OpKernel {
 public:
  explicit ExpandDimsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    int dim = ctx->input(1).flat<int>()(0);
    OP_REQUIRES(
        ctx, (dim >= -1 - ctx->input(0).dims() && dim <= ctx->input(0).dims()),
        errors::InvalidArgument("Tried to expand dim index ", dim,
                                " for tensor with ", ctx->input(0).dims(),
                                " dimensions."));

    auto existing_dims = ctx->input(0).shape().dim_sizes();
    std::vector<int64> new_shape(existing_dims.size());
    for (size_t i = 0; i < new_shape.size(); ++i) {
      new_shape[i] = existing_dims[i];
    }

    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    if (dim < 0) {
      dim += existing_dims.size() + 1;
    }

    // Clamp to the end if needed.
    dim = std::min<int32>(dim, existing_dims.size());
    new_shape.emplace(new_shape.begin() + dim, 1);
    const TensorShape output_shape(new_shape);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {0}, &output));
    if (!output->CopyFrom(ctx->input(0), output_shape)) {
      // This should never happen, since the sizes of the input and output
      // should always be the same (we only expand the dimension with 1).
      ctx->SetStatus(
          errors::Internal("Could not expand dimension with input shape ",
                           ctx->input(0).shape().DebugString(),
                           " and output shape ", output_shape.DebugString()));
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ExpandDims").Device(DEVICE_CPU).HostMemory("dim"),
                        ExpandDimsOp);

#define REGISTER_GPU_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")             \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("dim"),        \
                          ExpandDimsOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp);

class SqueezeOp : public OpKernel {
 public:
  explicit SqueezeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<int32> squeeze_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("squeeze_dims", &squeeze_dims));
    squeeze_dims_.insert(squeeze_dims.begin(), squeeze_dims.end());
  }

  void Compute(OpKernelContext* ctx) override {
    auto existing_dims = ctx->input(0).shape().dim_sizes();
    std::vector<int64> new_shape;

    std::unordered_set<int32> wrapped_squeeze_dims;
    wrapped_squeeze_dims.reserve(squeeze_dims_.size());
    // Validate squeeze dims against the input.
    for (int32 dim : squeeze_dims_) {
      OP_REQUIRES(
          ctx, (dim >= -ctx->input(0).dims() && dim < ctx->input(0).dims()),
          errors::InvalidArgument("Tried to squeeze dim index ", dim,
                                  " for tensor with ", ctx->input(0).dims(),
                                  " dimensions."));
      // If dim is < 0, we wrap around (-1 means the last element).
      if (dim < 0) {
        dim = existing_dims.size() + dim;
      }

      wrapped_squeeze_dims.insert(dim);
    }

    for (size_t i = 0; i < existing_dims.size(); ++i) {
      auto existing_dim = existing_dims[i];

      // If squeeze_set is non-empty, only squeeze those dimensions.
      if (!wrapped_squeeze_dims.empty()) {
        if (wrapped_squeeze_dims.count(i) > 0) {
          OP_REQUIRES(ctx, existing_dim == 1,
                      errors::InvalidArgument("Tried to explicitly squeeze "
                                              "dimension ",
                                              i, " but dimension was not 1: ",
                                              existing_dim));
        } else {
          // This dimension is not being squeezed.
          new_shape.push_back(existing_dim);
        }
      } else {
        // Copy over all non-1-length dimensions.
        if (existing_dim != 1) {
          new_shape.push_back(existing_dim);
        }
      }
    }

    const TensorShape output_shape(new_shape);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {0}, &output));
    if (!output->CopyFrom(ctx->input(0), output_shape)) {
      // This should never happen, since the sizes of the input and
      // output should always be the same.
      ctx->SetStatus(errors::Internal("Could not squeeze input with shape ",
                                      ctx->input(0).shape().DebugString(),
                                      " and output shape ",
                                      output_shape.DebugString()));
    }
  }

 private:
  std::unordered_set<int32> squeeze_dims_;
};

REGISTER_KERNEL_BUILDER(Name("Squeeze").Device(DEVICE_CPU), SqueezeOp);

#define REGISTER_GPU_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Squeeze").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SqueezeOp);
TF_CALL_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
