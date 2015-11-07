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
// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

<<<<<<< HEAD
#include "tensorflow/core/kernels/relu_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
=======
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/relu_op.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
<<<<<<< HEAD
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

#define REGISTER_RELU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      ReluOp<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      ReluGradOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),         \
      Relu6Op<CPUDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<CPUDevice, type>)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyRelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      LeakyReluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_RELU_KERNELS);
#undef REGISTER_RELU_KERNELS

#define REGISTER_ELU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Elu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      EluOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("EluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      EluGradOp<CPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Selu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SeluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SeluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SeluGradOp<CPUDevice, type>)

// Elu and Selu only make sense with float or double.
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ELU_KERNELS);
#undef REGISTER_ELU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void Relu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Relu<GPUDevice, T>;                                   \
                                                                               \
  template <>                                                                  \
  void ReluGrad<GPUDevice, T>::operator()(                                     \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct ReluGrad<GPUDevice, T>;                               \
                                                                               \
  template <>                                                                  \
  void Relu6<GPUDevice, T>::operator()(                                        \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Relu6<GPUDevice, T>;                                  \
                                                                               \
  template <>                                                                  \
  void Relu6Grad<GPUDevice, T>::operator()(                                    \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct Relu6Grad<GPUDevice, T>;                              \
                                                                               \
  template <>                                                                  \
  void LeakyRelu<GPUDevice, T>::operator()(                                    \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features, T alpha,   \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct LeakyRelu<GPUDevice, T>;                              \
                                                                               \
  template <>                                                                  \
  void LeakyReluGrad<GPUDevice, T>::operator()(                                \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features, T alpha,                       \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct LeakyReluGrad<GPUDevice, T>;                          \
                                                                               \
  template <>                                                                  \
  void Elu<GPUDevice, T>::operator()(const GPUDevice& d,                       \
                                     typename TTypes<T>::ConstTensor features, \
                                     typename TTypes<T>::Tensor activations);  \
  extern template struct Elu<GPUDevice, T>;                                    \
                                                                               \
  template <>                                                                  \
  void EluGrad<GPUDevice, T>::operator()(                                      \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor activations,                             \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct EluGrad<GPUDevice, T>;                                \
                                                                               \
  template <>                                                                  \
  void Selu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Selu<GPUDevice, T>;                                   \
                                                                               \
  template <>                                                                  \
  void SeluGrad<GPUDevice, T>::operator()(                                     \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor activations,                             \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct SeluGrad<GPUDevice, T>;

#if GOOGLE_CUDA
// TODO(rocm) : qint8 datatype currently not supported on the ROCm platform
template <>
void Relu<GPUDevice, qint8>::operator()(
    const GPUDevice& d, typename TTypes<qint8>::ConstTensor features,
    typename TTypes<qint8>::Tensor activations);
extern template struct Relu<GPUDevice, qint8>;
#endif  // GOOGLE_CUDA
=======

template <typename Device, typename T>
class ReluOp : public UnaryElementWiseOp<T, ReluOp<Device, T>> {
 public:
  using UnaryElementWiseOp<T, ReluOp<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Relu<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class Relu6Op : public UnaryElementWiseOp<T, Relu6Op<Device, T>> {
 public:
  using UnaryElementWiseOp<T, Relu6Op<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Relu6<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class ReluGradOp : public BinaryElementWiseOp<T, ReluGradOp<Device, T>> {
 public:
  using BinaryElementWiseOp<T, ReluGradOp<Device, T>>::BinaryElementWiseOp;

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to ReluOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OP_REQUIRES(context, a.IsSameSize(g),
                errors::InvalidArgument("g and a must be the same size"));
    functor::ReluGrad<Device, T> functor;
    functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class Relu6GradOp : public BinaryElementWiseOp<T, Relu6GradOp<Device, T>> {
 public:
  using BinaryElementWiseOp<T, Relu6GradOp<Device, T>>::BinaryElementWiseOp;

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to Relu6Op()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OP_REQUIRES(context, a.IsSameSize(g),
                errors::InvalidArgument("g and a must be the same size"));
    functor::Relu6Grad<Device, T> functor;
    functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
            output->flat<T>());
  }
};

#define REGISTER_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      ReluOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      Relu6Op<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      ReluGradOp<CPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      Relu6GradOp<CPUDevice, type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                          \
  template <>                                                        \
  void Relu<GPUDevice, T>::operator()(                               \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,  \
      typename TTypes<T>::Tensor activations);                       \
  extern template struct Relu<GPUDevice, T>;                         \
                                                                     \
  template <>                                                        \
  void ReluGrad<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features,                      \
      typename TTypes<T>::Tensor backprops);                         \
                                                                     \
  extern template struct ReluGrad<GPUDevice, T>;                     \
  template <>                                                        \
  void Relu6<GPUDevice, T>::operator()(                              \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,  \
      typename TTypes<T>::Tensor activations);                       \
  extern template struct Relu6<GPUDevice, T>;                        \
                                                                     \
  template <>                                                        \
  void Relu6Grad<GPUDevice, T>::operator()(                          \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features,                      \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct Relu6Grad<GPUDevice, T>;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
<<<<<<< HEAD
#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
      ReluOp<GPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      ReluGradOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6").Device(DEVICE_GPU).TypeConstraint<type>("T"),         \
      Relu6Op<GPUDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6Grad").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyRelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      LeakyReluOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Elu").Device(DEVICE_GPU).TypeConstraint<type>("T"),           \
      EluOp<GPUDevice, type>);                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("EluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      EluGradOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Selu").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
      SeluOp<GPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      SeluGradOp<GPUDevice, type>)
=======
#define REGISTER_GPU_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      ReluOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      Relu6Op<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      ReluGradOp<GPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6Grad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      Relu6GradOp<GPUDevice, type>)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

<<<<<<< HEAD
#if GOOGLE_CUDA
template <typename Device>
class ReluOp<Device, qint8>
    : public UnaryElementWiseOp<qint8, ReluOp<Device, qint8>> {
 public:
  using UnaryElementWiseOp<qint8, ReluOp<Device, qint8>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    auto flat_input = input.flat<qint8>();
    OP_REQUIRES(context, (flat_input.size() % 4) == 0,
                errors::InvalidArgument(
                    "Tensor size must be a multiple of 4 for Relu<qint8>. Got ",
                    flat_input.size()));
    functor::Relu<Device, qint8> func;
    func(context->eigen_device<Device>(), flat_input, output->flat<qint8>());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Relu").Device(DEVICE_GPU).TypeConstraint<qint8>("T"),
    ReluOp<GPUDevice, qint8>);

#endif  // GOOGLE_CUDA
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
// Registration of the GPU implementations.
#define REGISTER_SYCL_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Relu").Device(DEVICE_SYCL).TypeConstraint<type>("T"),          \
      ReluOp<SYCLDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ReluGrad").Device(DEVICE_SYCL).TypeConstraint<type>("T"),      \
      ReluGradOp<SYCLDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Relu6").Device(DEVICE_SYCL).TypeConstraint<type>("T"),         \
      Relu6Op<SYCLDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Relu6Grad").Device(DEVICE_SYCL).TypeConstraint<type>("T"),     \
      Relu6GradOp<SYCLDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("LeakyRelu").Device(DEVICE_SYCL).TypeConstraint<type>("T"),     \
      LeakyReluOp<SYCLDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("LeakyReluGrad").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      LeakyReluGradOp<SYCLDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Elu").Device(DEVICE_SYCL).TypeConstraint<type>("T"),           \
      EluOp<SYCLDevice, type>);                                            \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("EluGrad").Device(DEVICE_SYCL).TypeConstraint<type>("T"),       \
      EluGradOp<SYCLDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Selu").Device(DEVICE_SYCL).TypeConstraint<type>("T"),          \
      SeluOp<SYCLDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("SeluGrad").Device(DEVICE_SYCL).TypeConstraint<type>("T"),      \
      SeluGradOp<SYCLDevice, type>)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL_KERNELS);
#undef REGISTER_SYCL_KERNELS
#endif  // TENSORFLOW_USE_SYCL
=======
#endif  // GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace tensorflow
