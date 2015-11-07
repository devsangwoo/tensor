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

#include <limits>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
=======
// See docs in ../ops/array_ops.cc.

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/concat_op.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/public/status.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
<<<<<<< HEAD
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL
=======
typedef Eigen::GpuDevice GPUDevice;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// --------------------------------------------------------------------------
template <typename Device, typename T>
class PackOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

<<<<<<< HEAD
  explicit PackOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }
=======
  explicit PackOp(OpKernelConstruction* c) : OpKernel(c) {}
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  void Compute(OpKernelContext* c) override {
    OpInputList values;
    OP_REQUIRES_OK(c, c->input_list("values", &values));
    const int num = values.size();

    // Verify that all input shapes match
    for (int i = 1; i < num; i++) {
      OP_REQUIRES(c, values[0].shape().IsSameSize(values[i].shape()),
                  errors::InvalidArgument(
                      "Shapes of all inputs must match: values[0].shape = ",
<<<<<<< HEAD
                      values[0].shape().DebugString(), " != values[", i,
                      "].shape = ", values[i].shape().DebugString()));
    }

    int expanded_num_dims = values[0].dims() + 1;
    int axis = axis_;
    if (axis < 0) axis += expanded_num_dims;

    OP_REQUIRES(c, 0 <= axis && axis < expanded_num_dims,
                errors::InvalidArgument("axis = ", axis_, " not in [",
                                        -expanded_num_dims, ", ",
                                        expanded_num_dims, ")"));

    TensorShape output_shape(values[0].shape());
    output_shape.InsertDim(axis, num);
=======
                      values[0].shape().ShortDebugString(), " != values[", i,
                      "].shape = ", values[i].shape().ShortDebugString()));
    }

    TensorShape output_shape(values[0].shape());
    output_shape.InsertDim(0, num);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    // In the num = 1 case, just reshape the input
    if (num == 1) {
      Tensor output;
      CHECK(output.CopyFrom(values[0], output_shape));
      c->set_output(0, output);
      return;
    }

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));

<<<<<<< HEAD
    int64 before_dim = 1;
    for (int i = 0; i < axis; ++i) {
      before_dim *= output_shape.dim_size(i);
    }

    int64 after_dim = 1;
    for (int i = axis + 1; i < output_shape.dims(); ++i) {
      after_dim *= output_shape.dim_size(i);
    }

    const int64 axis_dim = output_shape.dim_size(axis);

    const int64 output_size = output->NumElements();
    if (output_size > 0) {
      auto output_flat =
          output->shaped<T, 2>({before_dim, after_dim * axis_dim});
=======
    const int output_size = output->NumElements();
    if (output_size > 0) {
      auto output_flat = output->shaped<T, 2>({1, output_size});
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

      // Except for shapes, pack is a special case of concat, so we reuse the
      // same computational kernels.
      ConstMatrixVector inputs_flat;
      inputs_flat.reserve(num);
      for (int i = 0; i < num; ++i) {
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
<<<<<<< HEAD
            values[i].shaped<T, 2>({before_dim, after_dim})));
      }
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      if (std::is_same<Device, GPUDevice>::value) {
        ConcatGPU<T>(c, inputs_flat, output, &output_flat);
        return;
      }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#ifdef TENSORFLOW_USE_SYCL
      if (std::is_same<Device, SYCLDevice>::value) {
        ConcatSYCL<T>(c->eigen_sycl_device(), inputs_flat, &output_flat);
        return;
      }
#endif  // TENSORFLOW_USE_SYCL
      ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
    }
  }

 private:
  int axis_;
=======
            values[i].shaped<T, 2>({1, values[i].NumElements()})));
      }
      if (std::is_same<Device, GPUDevice>::value) {
        ConcatGPU<T>(c->eigen_gpu_device(), inputs_flat, &output_flat);
      } else {
        ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
      }
    }
  }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
};

#define REGISTER_PACK(type)                                      \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Pack").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      PackOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_PACK);
<<<<<<< HEAD
TF_CALL_QUANTIZED_TYPES(REGISTER_PACK);

#if defined(IS_MOBILE_PLATFORM) && !defined(SUPPORT_SELECTIVE_REGISTRATION)
// Primarily used for SavedModel support on mobile.
REGISTER_PACK(tstring);
#endif  // defined(IS_MOBILE_PLATFORM) &&
        // !defined(SUPPORT_SELECTIVE_REGISTRATION)

#undef REGISTER_PACK

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
=======
REGISTER_PACK(quint8);
REGISTER_PACK(qint8);
REGISTER_PACK(qint32);
REGISTER_PACK(bfloat16);

#undef REGISTER_PACK

#if GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define REGISTER_GPU(type)                                       \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Pack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PackOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
<<<<<<< HEAD
TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_int16(REGISTER_GPU);
TF_CALL_bool(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Pack")
                            .Device(DEVICE_GPU)
                            .HostMemory("values")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PackOp<CPUDevice, int32>);

<<<<<<< HEAD
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Pack").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      PackOp<SYCLDevice, type>)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL);
REGISTER_KERNEL_BUILDER(Name("Pack")
                            .Device(DEVICE_SYCL)
                            .HostMemory("values")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PackOp<CPUDevice, int32>);
#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL
=======
#endif  // GOOGLE_CUDA

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
