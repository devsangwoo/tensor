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
#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

<<<<<<< HEAD
#define REGISTER_CPU_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tidx"),                                      \
      ReductionOp<CPUDevice, type, int32, Eigen::internal::SumReducer<type>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tidx"),                                      \
      ReductionOp<CPUDevice, type, int64, Eigen::internal::SumReducer<type>>);
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tidx")                                       \
          .HostMemory("reduction_indices"),                                    \
      ReductionOp<GPUDevice, type, int32, Eigen::internal::SumReducer<type>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tidx")                                       \
          .HostMemory("reduction_indices"),                                    \
      ReductionOp<GPUDevice, type, int64, Eigen::internal::SumReducer<type>>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
#if GOOGLE_CUDA
TF_CALL_complex64(REGISTER_GPU_KERNELS);
TF_CALL_complex128(REGISTER_GPU_KERNELS);
#endif
#undef REGISTER_GPU_KERNELS

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int32>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int32, Eigen::internal::SumReducer<int32>>);
REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int64>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int64, Eigen::internal::SumReducer<int32>>);

#endif

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(Name("Sum")                                      \
                              .Device(DEVICE_SYCL)                         \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int32>("Tidx")               \
                              .HostMemory("reduction_indices"),            \
                          ReductionOp<SYCLDevice, type, int32,             \
                                      Eigen::internal::SumReducer<type>>); \
  REGISTER_KERNEL_BUILDER(Name("Sum")                                      \
                              .Device(DEVICE_SYCL)                         \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int64>("Tidx")               \
                              .HostMemory("reduction_indices"),            \
                          ReductionOp<SYCLDevice, type, int64,             \
                                      Eigen::internal::SumReducer<type>>);
REGISTER_SYCL_KERNELS(float);
REGISTER_SYCL_KERNELS(double);

REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_SYCL)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int32>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int32, Eigen::internal::SumReducer<int32>>);
REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_SYCL)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int64>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int64, Eigen::internal::SumReducer<int32>>);
#undef REGISTER_SYCL_KERNELS
#endif  // TENSORFLOW_USE_SYCL

=======
#define REGISTER_CPU_KERNELS(type)                              \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Sum").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ReductionOp<CPUDevice, type, Eigen::internal::SumReducer<type>>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

// NOTE: We should have mean(complex64,int32), too. But that needs to
// change Eigen::internal::MeanReducer to cast int to complex<float>.
// We don't see immediate need of mean(complex64,int32) anyway.
REGISTER_KERNEL_BUILDER(
    Name("Sum").Device(DEVICE_CPU).TypeConstraint<complex64>("T"),
    ReductionOp<CPUDevice, complex64, Eigen::internal::SumReducer<complex64>>);

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNELS(type)          \
  REGISTER_KERNEL_BUILDER(                  \
      Name("Sum")                           \
          .Device(DEVICE_GPU)               \
          .TypeConstraint<type>("T")        \
          .HostMemory("reduction_indices"), \
      ReductionOp<GPUDevice, type, Eigen::internal::SumReducer<type>>);
REGISTER_GPU_KERNELS(float);
#undef REGISTER_GPU_KERNELS

REGISTER_KERNEL_BUILDER(
    Name("Sum").Device(DEVICE_GPU).TypeConstraint<complex64>("T"),
    ReductionOp<GPUDevice, complex64, Eigen::internal::SumReducer<complex64>>);

#endif

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
