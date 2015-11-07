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
// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

<<<<<<< HEAD
#include <numeric>

#include "tensorflow/core/kernels/aggregate_ops.h"
#include "tensorflow/core/kernels/aggregate_ops_cpu.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"

=======
#include "tensorflow/core/kernels/aggregate_ops.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"

#include "tensorflow/core/platform/logging.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
<<<<<<< HEAD
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

template <typename Device, typename T>
class AddNOp : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    const Tensor& input0 = ctx->input(0);
<<<<<<< HEAD
    const int num = ctx->num_inputs();

    if (num == 1) {
      ctx->set_output(0, input0);
      return;
    }

    // Try to forward and accumulate the result in one of the input buffers.
    int reused_input = -1;
    gtl::InlinedVector<int, 8> input_indices(num);
    std::iota(input_indices.begin(), input_indices.end(), 0);
    Tensor* output = nullptr;
    for (int input_idx = 0; input_idx < num; ++input_idx) {
      if (ctx->forward_input_to_output_with_shape(input_idx, 0, input0.shape(),
                                                  &output)) {
        reused_input = input_idx;
        break;
      }
    }
    if (reused_input == -1) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input0.shape(), &output));
    } else if (reused_input > 0) {
      // Move the forwarded buffer to the front so we don't double count
      // anything if there are more than 8 inputs.
      input_indices[0] = reused_input;
      input_indices[reused_input] = 0;
    }
    auto To = output->flat<T>();

#define I(IDX) ctx->input(input_indices[IDX]).flat<T>()

#if defined(__ANDROID_TYPES_SLIM__)
    // On Android by default,we only support additions of two arguments, so we
=======
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input0.shape(), &output));
    auto To = output->flat<T>();

    const int num = ctx->num_inputs();
    if (num == 1) {
      *output = input0;
      return;
    }

#define I(IDX) ctx->input(IDX).flat<T>()

#if defined(PLATFORM_POSIX_ANDROID) || defined(PLATFORM_GOOGLE_ANDROID)
    // On Android, we only support additions of two arguments, so we
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    // can reduce the number of template instantiations.
    OP_REQUIRES(ctx, num == 2,
                errors::InvalidArgument("Only additions of two arguments "
                                        "supported. Num inputs: ",
                                        num));
    functor::Add2Functor<Device, T> functor2;
    functor2(ctx->template eigen_device<Device>(), To, I(0), I(1));
#else
    static const int kWidth = 8;
    int r = num % kWidth;

    switch (r) {
      case 2: {
        functor::Add2Functor<Device, T> functor2;
        functor2(ctx->template eigen_device<Device>(), To, I(0), I(1));
        break;
      }
      case 3: {
        functor::Add3Functor<Device, T> functor3;
        functor3(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2));
        break;
      }
      case 4: {
        functor::Add4Functor<Device, T> functor4;
        functor4(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3));
        break;
      }
      case 5: {
        functor::Add5Functor<Device, T> functor5;
        functor5(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4));
        break;
      }
      case 6: {
        functor::Add6Functor<Device, T> functor6;
        functor6(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5));
        break;
      }
      case 7: {
        functor::Add7Functor<Device, T> functor7;
        functor7(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6));
        break;
      }
      case 0: {
        functor::Add8Functor<Device, T> functor8;
        functor8(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6), I(7));
        r = 8;
        break;
      }
      case 1: {
        functor::Add9Functor<Device, T> functor9;
        functor9(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6), I(7), I(8));
        r = 9;
        break;
      }
    }

    for (; r < num; r += kWidth) {
      functor::Add8pFunctor<Device, T> functor8p;
      functor8p(ctx->template eigen_device<Device>(), To, I(r), I(r + 1),
                I(r + 2), I(r + 3), I(r + 4), I(r + 5), I(r + 6), I(r + 7));
    }
<<<<<<< HEAD
#endif  // defined(__ANDROID_TYPES_SLIM__)
=======
#endif  // defined(PLATFORM_POSIX_ANDROID) || defined(PLATFORM_GOOGLE_ANDROID)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#undef I
  }
};

<<<<<<< HEAD
template <typename Device>
class AddNOp<Device, Variant> : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    const Tensor& input0 = ctx->input(0);
    const int num = ctx->num_inputs();

    if (num == 1) {
      ctx->set_output(0, input0);
      return;
    }

    for (int i = 0; i < num; ++i) {
      // Step 1: ensure unary variants.
      OP_REQUIRES(
          ctx, ctx->input(i).dims() == 0,
          errors::InvalidArgument(
              "AddN of non-scalar Tensor with dtype=DT_VARIANT is not "
              "supported; inputs[",
              i, " has shape: ", ctx->input(i).shape().DebugString(), "."));
    }

    // Step 2: attempt to add using
    //   BinaryOpVariants(ADD_VARIANT_BINARY_OP, ...)
    //   For the output create a default-constructed variant object.
    // TODO(ebrevdo): Perform summation in a tree-structure.
    Tensor out(cpu_allocator(), DT_VARIANT, TensorShape({}));
    Variant* v_out = &(out.scalar<Variant>()());
    OP_REQUIRES_OK(
        ctx, BinaryOpVariants<Device>(
                 ctx, ADD_VARIANT_BINARY_OP, ctx->input(0).scalar<Variant>()(),
                 ctx->input(1).scalar<Variant>()(), v_out));
    for (int i = 2; i < num; ++i) {
      const Variant tmp = std::move(*v_out);
      const Variant& inp = ctx->input(i).scalar<Variant>()();
      OP_REQUIRES_OK(ctx, BinaryOpVariants<Device>(ctx, ADD_VARIANT_BINARY_OP,
                                                   inp, tmp, v_out));
    }
    ctx->set_output(0, out);
  }
};

=======
// Partial specializations for a CPUDevice, that uses the Eigen implementation
// from AddNEigenImpl.
namespace functor {
template <typename T>
struct Add2Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2) {
    Add2EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2);
  }
};
template <typename T>
struct Add3Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3) {
    Add3EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3);
  }
};
template <typename T>
struct Add4Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4) {
    Add4EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4);
  }
};
template <typename T>
struct Add5Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5) {
    Add5EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5);
  }
};
template <typename T>
struct Add6Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6) {
    Add6EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6);
  }
};
template <typename T>
struct Add7Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6,
                  typename TTypes<T>::ConstFlat in7) {
    Add7EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7);
  }
};

template <typename T>
struct Add8Functor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8);
  }
};

template <typename T>
struct Add8pFunctor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8pEigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                          in7, in8);
  }
};

template <typename T>
struct Add9Functor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8,
      typename TTypes<T>::ConstFlat in9) {
    Add9EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8, in9);
  }
};

}  // namespace functor

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#define REGISTER_ADDN(type, dev)                                   \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AddN").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      AddNOp<dev##Device, type>)

#define REGISTER_ADDN_CPU(type) REGISTER_ADDN(type, CPU)

TF_CALL_NUMBER_TYPES(REGISTER_ADDN_CPU);
<<<<<<< HEAD
REGISTER_ADDN_CPU(Variant);

#undef REGISTER_ADDN_CPU

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define REGISTER_ADDN_GPU(type) REGISTER_ADDN(type, GPU)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ADDN_GPU);
TF_CALL_int64(REGISTER_ADDN_GPU);
TF_CALL_complex64(REGISTER_ADDN_GPU);
TF_CALL_complex128(REGISTER_ADDN_GPU);
TF_CALL_variant(REGISTER_ADDN_GPU);
#undef REGISTER_ADDN_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("AddN")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("inputs")
                            .HostMemory("sum"),
                        AddNOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
REGISTER_ADDN(float, SYCL);
REGISTER_ADDN(double, SYCL);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("AddN")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("inputs")
                            .HostMemory("sum"),
                        AddNOp<CPUDevice, int32>);
#endif  // TENSORFLOW_USE_SYCL
=======
#undef REGISTER_ADDN_CPU

#if GOOGLE_CUDA
REGISTER_ADDN(float, GPU);
#endif  // GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#undef REGISTER_ADDN

}  // namespace tensorflow
