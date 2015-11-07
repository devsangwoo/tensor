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
#include "tensorflow/core/kernels/l2loss_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
=======
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/l2loss_op.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
<<<<<<< HEAD

template <typename T>
class L2LossOp<CPUDevice, T> : public OpKernel {
=======
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class L2LossOp : public OpKernel {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
 public:
  explicit L2LossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // The input tensor can be of any number of dimensions, even though it's
    // 2D in most typical applications.
    const Tensor& input = context->input(0);
    // The output is a single number.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
<<<<<<< HEAD
    const CPUDevice& d = context->eigen_device<CPUDevice>();
    output->scalar<T>().device(d) =
        (input.flat<T>().square() * static_cast<T>(0.5)).sum();
=======
    functor::L2Loss<Device, T>()(context->eigen_device<Device>(),
                                 input.flat<T>(), output->scalar<T>());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
};

#define REGISTER_KERNEL(T)                                      \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("L2Loss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      L2LossOp<CPUDevice, T>);

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
<<<<<<< HEAD
REGISTER_KERNEL(Eigen::half);
#ifdef ENABLE_INTEL_MKL_BFLOAT16
// Since Eigen backend does not support bfloat16 ops, we are selectively
// enabling them for MKL backend.
REGISTER_KERNEL(bfloat16);
#endif
#undef REGISTER_KERNEL

=======
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void L2Loss<GPUDevice, T>::operator()(const GPUDevice& d,                    \
                                        typename TTypes<T>::ConstTensor input, \
                                        typename TTypes<T>::Scalar output);    \
  extern template struct L2Loss<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("L2Loss").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      L2LossOp<GPUDevice, T>);

REGISTER_GPU_KERNEL(float);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
