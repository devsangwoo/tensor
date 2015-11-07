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

<<<<<<< HEAD
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
=======
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/tile_ops.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
<<<<<<< HEAD
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

// Forward declarations of functors that will be defined in tile_ops_impl.h
namespace functor {
template <typename Device, typename T, typename Tmultiple>
struct Tile {
  void operator()(const Device& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<Tmultiple> broadcast_array) const;
};

template <typename Device, typename T, int NDIM>
struct TileGrad {
  void operator()(const Device& d, typename TTypes<T, NDIM>::Tensor out,
                  typename TTypes<T, NDIM>::ConstTensor in,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes,
                  bool first) const;
};

template <typename Device, typename T>
struct TileGrad<Device, T, 0> {
  void operator()(const Device& d, typename TTypes<T, 0>::Tensor out,
                  typename TTypes<T, 0>::ConstTensor in,
                  const Eigen::DSizes<Eigen::DenseIndex, 0>&,
                  const Eigen::DSizes<Eigen::DenseIndex, 0>&, bool first) const;
};

template <typename Device, typename T, int NDIM, int REDUCEDNDIM>
struct ReduceAndReshape {
  void operator()(
      const Device& d, typename TTypes<T, NDIM>::Tensor out,
      typename TTypes<T, NDIM>::ConstTensor in,
      const Eigen::DSizes<Eigen::DenseIndex, REDUCEDNDIM>& reduce_dim,
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& reshape_dim) const;
};

// Explicit instantiations are defined in tile_ops_{cpu,gpu}_impl.*,
// below are their declarations.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
extern template struct Tile<GPUDevice, bool, int32>;
extern template struct Tile<GPUDevice, bool, int64>;
extern template struct Tile<GPUDevice, float, int32>;
extern template struct Tile<GPUDevice, float, int64>;
extern template struct Tile<GPUDevice, double, int32>;
extern template struct Tile<GPUDevice, double, int64>;
extern template struct Tile<GPUDevice, complex64, int32>;
extern template struct Tile<GPUDevice, complex64, int64>;
extern template struct Tile<GPUDevice, complex128, int32>;
extern template struct Tile<GPUDevice, complex128, int64>;
extern template struct Tile<GPUDevice, Eigen::half, int32>;
extern template struct Tile<GPUDevice, Eigen::half, int64>;
extern template struct Tile<GPUDevice, int16, int32>;
extern template struct Tile<GPUDevice, int16, int64>;
extern template struct Tile<GPUDevice, int32, int32>;
extern template struct Tile<GPUDevice, int32, int64>;
extern template struct Tile<GPUDevice, int64, int32>;
extern template struct Tile<GPUDevice, int64, int64>;
#define DECLARE_CUDA_DIM(T, NDIM)                      \
  extern template struct TileGrad<GPUDevice, T, NDIM>; \
  extern template struct ReduceAndReshape<GPUDevice, T, NDIM, 1>
#else  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define DECLARE_CUDA_DIM(T, NDIM)
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
#define DECLARE_TYPE(T)                              \
  extern template struct Tile<SYCLDevice, T, int32>; \
  extern template struct Tile<SYCLDevice, T, int64>;
TF_CALL_bool(DECLARE_TYPE);
TF_CALL_float(DECLARE_TYPE);
TF_CALL_bfloat16(DECLARE_TYPE);
TF_CALL_double(DECLARE_TYPE);
TF_CALL_uint8(DECLARE_TYPE);
TF_CALL_int32(DECLARE_TYPE);
TF_CALL_int16(DECLARE_TYPE);
TF_CALL_int64(DECLARE_TYPE);
#undef DECLARE_TYPE
#define DECLARE_SYCL_DIM(T, NDIM)                       \
  extern template struct TileGrad<SYCLDevice, T, NDIM>; \
  extern template struct ReduceAndReshape<SYCLDevice, T, NDIM, 1>
#else  // TENSORFLOW_USE_SYCL
#define DECLARE_SYCL_DIM(T, NDIM)
#endif  // TENSORFLOW_USE_SYCL

#define DECLARE_TYPE(T)                             \
  extern template struct Tile<CPUDevice, T, int32>; \
  extern template struct Tile<CPUDevice, T, int64>;
TF_CALL_bool(DECLARE_TYPE);
TF_CALL_float(DECLARE_TYPE);
TF_CALL_bfloat16(DECLARE_TYPE);
TF_CALL_double(DECLARE_TYPE);
TF_CALL_uint8(DECLARE_TYPE);
TF_CALL_int32(DECLARE_TYPE);
TF_CALL_int16(DECLARE_TYPE);
TF_CALL_int64(DECLARE_TYPE);
TF_CALL_half(DECLARE_TYPE);
TF_CALL_complex64(DECLARE_TYPE);
TF_CALL_complex128(DECLARE_TYPE);
TF_CALL_tstring(DECLARE_TYPE);
#undef DECLARE_TYPE

#define DECLARE_DIM(T, NDIM)                           \
  DECLARE_CUDA_DIM(T, NDIM);                           \
  DECLARE_SYCL_DIM(T, NDIM);                           \
  extern template struct TileGrad<CPUDevice, T, NDIM>; \
  extern template struct ReduceAndReshape<CPUDevice, T, NDIM, 1>;

#define DECLARE_TYPE(T) \
  DECLARE_DIM(T, 1)     \
  DECLARE_DIM(T, 2)     \
  DECLARE_DIM(T, 3)     \
  DECLARE_DIM(T, 4)     \
  DECLARE_DIM(T, 5)     \
  DECLARE_DIM(T, 6)     \
  DECLARE_DIM(T, 7)
TF_CALL_float(DECLARE_TYPE);
TF_CALL_bfloat16(DECLARE_TYPE);
TF_CALL_double(DECLARE_TYPE);
TF_CALL_int16(DECLARE_TYPE);
TF_CALL_int32(DECLARE_TYPE);
TF_CALL_int64(DECLARE_TYPE);
TF_CALL_half(DECLARE_TYPE);
TF_CALL_complex64(DECLARE_TYPE);
TF_CALL_complex128(DECLARE_TYPE);
#undef DECLARE_TYPE

#undef DECLARE_DIM
#undef DECLARE_SYCL_DIM
#undef DECLARE_CUDA_DIM

}  // namespace functor

// --------------------------------------------------------------------------
template <typename Device, typename Tmultiples>
=======

// --------------------------------------------------------------------------
template <typename Device>
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
class TileOp : public OpKernel {
 public:
  explicit TileOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& multiples = context->input(1);

    OP_REQUIRES(
<<<<<<< HEAD
        context, IsLegacyVector(multiples.shape()),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples.shape().DebugString()));
=======
        context, TensorShapeUtils::IsLegacyVector(multiples.shape()),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples.shape().ShortDebugString()));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    OP_REQUIRES(context, input.dims() == multiples.NumElements(),
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input.dims(), " but got length ", multiples.dim_size(0)));
<<<<<<< HEAD
    const int input_dims = input.dims();

    // Eigen doesn't support scalars on the GPU, so handle 0-D specially
    if (input_dims == 0) {
      context->set_output(0, input);
      return;
    }

    const gtl::ArraySlice<Tmultiples> multiples_array(
        multiples.flat<Tmultiples>().data(), input_dims);
    TensorShape output_shape;
    for (int i = 0; i < input_dims; ++i) {
      OP_REQUIRES(
          context, multiples_array[i] >= 0,
          errors::InvalidArgument("Expected multiples[", i, "] >= 0, but got ",
                                  multiples_array[i]));
      output_shape.AddDim(input.dim_size(i) * multiples_array[i]);
    }
    if (output_shape == input.shape()) {
      context->set_output(0, input);
      return;
    }
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &result));

    // If there's no output, there's nothing to do.
    if (output_shape.num_elements() == 0) return;

#define HANDLE_TYPE(DT)                               \
  if (context->input(0).dtype() == DT) {              \
    HandleCase<DT>(context, multiples_array, result); \
    return;                                           \
  }

#define HANDLE_TYPE_NAME(T) HANDLE_TYPE(DataTypeToEnum<T>::value)

    // Invoke macro using TF_CALL_* so type-filtering for platform applies.
    TF_CALL_bool(HANDLE_TYPE_NAME);
    TF_CALL_bfloat16(HANDLE_TYPE_NAME);
    TF_CALL_float(HANDLE_TYPE_NAME);
    TF_CALL_double(HANDLE_TYPE_NAME);
    TF_CALL_uint8(HANDLE_TYPE_NAME);
    TF_CALL_int8(HANDLE_TYPE_NAME);
    TF_CALL_int32(HANDLE_TYPE_NAME);
    TF_CALL_int16(HANDLE_TYPE_NAME);
    TF_CALL_int64(HANDLE_TYPE_NAME);
    TF_CALL_half(HANDLE_TYPE_NAME);
    TF_CALL_tstring(HANDLE_TYPE_NAME);  // when DEVICE=CPUDevice.
    TF_CALL_complex64(HANDLE_TYPE_NAME);
    TF_CALL_complex128(HANDLE_TYPE_NAME);

#undef HANDLE_TYPE_NAME
#undef HANDLE_TYPE

    OP_REQUIRES(
        context, false,
        errors::Unimplemented(
            "TileOp : The input data type is not supported, DataType : ",
            DataTypeString(context->input(0).dtype()),
            ", Dimension : ", input_dims));
  }

 private:
  template <DataType DT>
  void HandleCaseImpl(OpKernelContext* context,
                      const gtl::ArraySlice<Tmultiples>& multiples_array,
                      Tensor* result) {
    typedef typename EnumToDataType<DT>::Type T;
    functor::Tile<Device, T, Tmultiples>()(context->eigen_device<Device>(),
                                           result, context->input(0),
                                           multiples_array);
  }

  template <DataType DT>
  void HandleCase(OpKernelContext* context,
                  const gtl::ArraySlice<Tmultiples>& multiples_array,
=======

    const int input_dims = input.dims();
    const gtl::ArraySlice<int32> multiples_array(multiples.flat<int32>().data(),
                                                 input_dims);

    TensorShape output_shape;
    for (int i = 0; i < input_dims; ++i) {
      OP_REQUIRES(
          context, multiples_array[i] > 0,
          errors::InvalidArgument("Expected multiples[", i, "] > 0, but got ",
                                  multiples_array[i]));
      output_shape.AddDim(input.dim_size(i) * multiples_array[i]);
    }
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &result));

#define HANDLE_DIM(DT, NDIM)                                   \
  if (context->input(0).dtype() == DT && input_dims == NDIM) { \
    HandleCase<DT, NDIM>(context, multiples_array, result);    \
    return;                                                    \
  }

#define HANDLE_TYPE(T) \
  HANDLE_DIM(T, 0)     \
  HANDLE_DIM(T, 1)     \
  HANDLE_DIM(T, 2)     \
  HANDLE_DIM(T, 3)     \
  HANDLE_DIM(T, 4)     \
  HANDLE_DIM(T, 5)

    HANDLE_TYPE(DT_BOOL);
    HANDLE_TYPE(DT_FLOAT);
    HANDLE_TYPE(DT_DOUBLE);
    HANDLE_TYPE(DT_UINT8);
    HANDLE_TYPE(DT_INT32);
    HANDLE_TYPE(DT_INT16);
    HANDLE_TYPE(DT_INT64);
    HANDLE_TYPE(DT_STRING);  // when DEVICE=CPUDevice.

#undef HANDLE_TYPE
#undef HANDLE_DIM

    OP_REQUIRES(context, false,
                errors::Unimplemented(
                    "TileOp : Unhandled input dimensions, DT : ",
                    context->input(0).dtype(), ", dims : ", input_dims));
  }

 private:
  template <DataType DT, int NDIM>
  void HandleCaseImpl(OpKernelContext* context,
                      const gtl::ArraySlice<int32>& multiples_array,
                      Tensor* result) {
    typedef typename EnumToDataType<DT>::Type T;
    Eigen::array<int32, NDIM> broadcast_array;
    for (int i = 0; i < NDIM; ++i) {
      broadcast_array[i] = multiples_array[i];
    }
    functor::Tile<Device, T, NDIM>()(
        context->eigen_device<Device>(), result->tensor<T, NDIM>(),
        context->input(0).tensor<T, NDIM>(), broadcast_array);
  }

  template <DataType DT, int NDIM>
  void HandleCase(OpKernelContext* context,
                  const gtl::ArraySlice<int32>& multiples_array,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                  Tensor* result);

  TF_DISALLOW_COPY_AND_ASSIGN(TileOp);
};

<<<<<<< HEAD
template <typename Device, typename Tmultiples>
template <DataType DT>
inline void TileOp<Device, Tmultiples>::HandleCase(
    OpKernelContext* context,
    const gtl::ArraySlice<Tmultiples>& multiples_array, Tensor* result) {
  // TODO(vrv): print out the device name if useful. Currently disabled to avoid
  // having to use RTTI.
  LOG(FATAL) << "TileOp: Invalid combination of Device, DT: "
             // << typeid(Device).name() << ", "
             << DataTypeString(DT);
}

#define HANDLE_CASE(device, dtype, Tmultiples)                              \
  template <>                                                               \
  template <>                                                               \
  void TileOp<device, Tmultiples>::HandleCase<dtype>(                       \
      OpKernelContext * context,                                            \
      const gtl::ArraySlice<Tmultiples>& multiples_array, Tensor* result) { \
    HandleCaseImpl<dtype>(context, multiples_array, result);                \
  }

#define HANDLE_TYPE_NAME_CPU(T)                            \
  HANDLE_CASE(CPUDevice, DataTypeToEnum<T>::value, int32); \
  HANDLE_CASE(CPUDevice, DataTypeToEnum<T>::value, int64);

#define HANDLE_TYPE_NAME_GPU(T)                            \
  HANDLE_CASE(GPUDevice, DataTypeToEnum<T>::value, int32); \
  HANDLE_CASE(GPUDevice, DataTypeToEnum<T>::value, int64);

#ifdef TENSORFLOW_USE_SYCL
#define HANDLE_TYPE_NAME_SYCL(T)                            \
  HANDLE_CASE(SYCLDevice, DataTypeToEnum<T>::value, int32); \
  HANDLE_CASE(SYCLDevice, DataTypeToEnum<T>::value, int64);
#endif  // TENSORFLOW_USE_SYCL

TF_CALL_bool(HANDLE_TYPE_NAME_CPU);
TF_CALL_float(HANDLE_TYPE_NAME_CPU);
TF_CALL_bfloat16(HANDLE_TYPE_NAME_CPU);
TF_CALL_double(HANDLE_TYPE_NAME_CPU);
TF_CALL_uint8(HANDLE_TYPE_NAME_CPU);
TF_CALL_int8(HANDLE_TYPE_NAME_CPU);
TF_CALL_int32(HANDLE_TYPE_NAME_CPU);
TF_CALL_int16(HANDLE_TYPE_NAME_CPU);
TF_CALL_int64(HANDLE_TYPE_NAME_CPU);
TF_CALL_half(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_CPU);
TF_CALL_tstring(HANDLE_TYPE_NAME_CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_bool(HANDLE_TYPE_NAME_GPU);
TF_CALL_float(HANDLE_TYPE_NAME_GPU);
TF_CALL_double(HANDLE_TYPE_NAME_GPU);
TF_CALL_int16(HANDLE_TYPE_NAME_GPU);
TF_CALL_int32(HANDLE_TYPE_NAME_GPU);
TF_CALL_int64(HANDLE_TYPE_NAME_GPU);
TF_CALL_half(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_GPU);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
TF_CALL_float(HANDLE_TYPE_NAME_SYCL);
TF_CALL_double(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int16(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int32(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int64(HANDLE_TYPE_NAME_SYCL);
#endif  // TENSORFLOW_USE_SYCL

#undef HANDLE_TYPE_NAME_CPU
#undef HANDLE_TYPE_NAME_GPU
#ifdef TENSORFLOW_USE_SYCL
#undef HANDLE_TYPE_NAME_SYCL
#endif  // TENSORFLOW_USE_SYCL
#undef HANDLE_CASE

// --------------------------------------------------------------------------
template <typename Device, typename Tmultiples>
=======
template <typename Device>
template <DataType DT, int NDIM>
inline void TileOp<Device>::HandleCase(
    OpKernelContext* context, const gtl::ArraySlice<int32>& multiples_array,
    Tensor* result) {
  LOG(FATAL) << "TileOp: Invalid combination of Device, DT and NDIM: "
             << typeid(Device).name() << ", " << DataTypeString(DT) << ", "
             << NDIM;
}

#define HANDLE_CASE(device, dtype, ndim)                               \
  template <>                                                          \
  template <>                                                          \
  void TileOp<device>::HandleCase<dtype, ndim>(                        \
      OpKernelContext * context,                                       \
      const gtl::ArraySlice<int32>& multiples_array, Tensor* result) { \
    HandleCaseImpl<dtype, ndim>(context, multiples_array, result);     \
  }

#define HANDLE_CASE_DIM_POSITIVE(device, dtype) \
  HANDLE_CASE(device, dtype, 1);                \
  HANDLE_CASE(device, dtype, 2);                \
  HANDLE_CASE(device, dtype, 3);                \
  HANDLE_CASE(device, dtype, 4);                \
  HANDLE_CASE(device, dtype, 5);

#define HANDLE_CASE_DIM(device, dtype) \
  HANDLE_CASE(device, dtype, 0);       \
  HANDLE_CASE_DIM_POSITIVE(device, dtype);

HANDLE_CASE_DIM(CPUDevice, DT_BOOL);
HANDLE_CASE_DIM(CPUDevice, DT_FLOAT);
HANDLE_CASE_DIM(CPUDevice, DT_DOUBLE);
HANDLE_CASE_DIM(CPUDevice, DT_UINT8);
HANDLE_CASE_DIM(CPUDevice, DT_INT32);
HANDLE_CASE_DIM(CPUDevice, DT_INT16);
HANDLE_CASE_DIM(CPUDevice, DT_INT64);
HANDLE_CASE_DIM(CPUDevice, DT_STRING);

#if GOOGLE_CUDA
// Eigen on GPU does not handle 0-dimension data types yet.
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_FLOAT);
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_DOUBLE);
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_INT16);
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_INT32);
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_INT64);
#endif  // GOOGLE_CUDA

#undef HANDLE_CASE_DIM_POSITIVE
#undef HANDLE_CASE_DIM
#undef HANDLE_CASE

// --------------------------------------------------------------------------
template <typename Device>
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
class TileGradientOp : public OpKernel {
 public:
  explicit TileGradientOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& multiples = context->input(1);
    OP_REQUIRES(
<<<<<<< HEAD
        context, IsLegacyVector(multiples.shape()),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples.shape().DebugString()));
=======
        context, TensorShapeUtils::IsLegacyVector(multiples.shape()),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples.shape().ShortDebugString()));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    OP_REQUIRES(context, input.dims() == multiples.NumElements(),
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input.dims(), " but got length ", multiples.dim_size(0)));

    const int input_dims = input.dims();
<<<<<<< HEAD

    // Eigen doesn't support scalars on the GPU, so handle 0-D specially
    if (input_dims == 0) {
      context->set_output(0, input);
      return;
    }

    const gtl::ArraySlice<Tmultiples> multiples_array(
        multiples.flat<Tmultiples>().data(), input_dims);
    TensorShape output_shape;
    std::vector<Tmultiples> input_dim_size_vec;
=======
    const gtl::ArraySlice<int32> multiples_array(multiples.flat<int32>().data(),
                                                 input_dims);

    TensorShape output_shape;
    std::vector<int32> input_dim_size_vec;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    for (int i = 0; i < input_dims; ++i) {
      OP_REQUIRES(
          context, multiples_array[i] > 0,
          errors::InvalidArgument("Expected multiples[", i, "] > 0, but got ",
                                  multiples_array[i]));
      OP_REQUIRES(context, input.dim_size(i) % multiples_array[i] == 0,
                  errors::InvalidArgument("Expected input_dim[", i,
                                          "] to be divisible by multiples[", i,
                                          "], but ", input.dim_size(i), " % ",
                                          multiples_array[i], " != 0"));
      output_shape.AddDim(input.dim_size(i) / multiples_array[i]);
      input_dim_size_vec.push_back(input.dim_size(i));
    }
<<<<<<< HEAD
    if (output_shape == input.shape()) {
      context->set_output(0, input);
      return;
    }
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &result));

#define HANDLE_DIM(DT, NDIM)                                           \
  if (context->input(0).dtype() == DT && input_dims == NDIM) {         \
    HandleCase<DT, NDIM>(context, input_dim_size_vec, multiples_array, \
                         result);                                      \
    return;                                                            \
  }

#define HANDLE_TYPE(T) \
<<<<<<< HEAD
=======
  HANDLE_DIM(T, 0)     \
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  HANDLE_DIM(T, 1)     \
  HANDLE_DIM(T, 2)     \
  HANDLE_DIM(T, 3)     \
  HANDLE_DIM(T, 4)     \
<<<<<<< HEAD
  HANDLE_DIM(T, 5)     \
  HANDLE_DIM(T, 6)     \
  HANDLE_DIM(T, 7)

#define HANDLE_TYPE_NAME(T) HANDLE_TYPE(DataTypeToEnum<T>::value)

    TF_CALL_float(HANDLE_TYPE_NAME);
    TF_CALL_double(HANDLE_TYPE_NAME);
    TF_CALL_int32(HANDLE_TYPE_NAME);
    TF_CALL_int16(HANDLE_TYPE_NAME);
    TF_CALL_int64(HANDLE_TYPE_NAME);
    TF_CALL_half(HANDLE_TYPE_NAME);
    TF_CALL_bfloat16(HANDLE_TYPE_NAME);
    TF_CALL_complex64(HANDLE_TYPE_NAME);
    TF_CALL_complex128(HANDLE_TYPE_NAME);

#undef HANDLE_TYPE_NAME
=======
  HANDLE_DIM(T, 5)

    HANDLE_TYPE(DT_FLOAT);
    HANDLE_TYPE(DT_DOUBLE);
    HANDLE_TYPE(DT_INT32);
    HANDLE_TYPE(DT_INT16);
    HANDLE_TYPE(DT_INT64);

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#undef HANDLE_TYPE
#undef HANDLE_DIM

    OP_REQUIRES(context, false,
<<<<<<< HEAD
                errors::Unimplemented("TileGradientOp : The input data type or "
                                      "dimension is not supported, DataType : ",
                                      DataTypeString(context->input(0).dtype()),
                                      ", Dimension : ", input_dims));
=======
                errors::Unimplemented(
                    "TileGradientOp : Unhandled input dimensions, DT : ",
                    context->input(0).dtype(), ", dims : ", input_dims));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

 private:
  template <DataType DT, int NDIM>
  void HandleCase(OpKernelContext* context,
<<<<<<< HEAD
                  const std::vector<Tmultiples>& input_dims,
                  const gtl::ArraySlice<Tmultiples>& multiples_array,
=======
                  const std::vector<int32>& input_dims,
                  const gtl::ArraySlice<int32>& multiples_array,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                  Tensor* result);

  template <DataType DT, int NDIM>
  void HandleCaseImpl(OpKernelContext* context,
<<<<<<< HEAD
                      const std::vector<Tmultiples>& input_dims,
                      const gtl::ArraySlice<Tmultiples>& multiples_array,
=======
                      const std::vector<int32>& input_dims,
                      const gtl::ArraySlice<int32>& multiples_array,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                      Tensor* result) {
    typedef typename EnumToDataType<DT>::Type T;

    bool reduction_only = true;
<<<<<<< HEAD
    std::vector<Tmultiples> reduction_dims;
=======
    std::vector<int> reduction_dims;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    for (int i = 0; i < NDIM; ++i) {
      if (input_dims[i] > multiples_array[i] && multiples_array[i] > 1) {
        reduction_only = false;
        break;
      } else {
        if (multiples_array[i] == input_dims[i]) {
          reduction_dims.push_back(i);
        }
      }
    }

    if (reduction_only) {
#define HANDLE_DIM(D)                                            \
  if (reduction_dims.size() == (D)) {                            \
    HandleReduce<T, NDIM, (D)>(context, reduction_dims, result); \
    return;                                                      \
  }
      // NOTE(keveman): Handling the most common case here.
      // Adding more cases here would require more templating and code
      // explosion. For instance, HANDLE_DIM(2) wouldn't make sense for NDIM=1.
<<<<<<< HEAD
      HANDLE_DIM(1);
=======
      HANDLE_DIM(NDIM > 0 ? 1 : 0);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// Fall through to the unoptimized version.
#undef HANDLE_DIM
    }

<<<<<<< HEAD
    Eigen::DSizes<Eigen::DenseIndex, NDIM> indices;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes;
=======
    Eigen::DSizes<ptrdiff_t, NDIM> indices;
    Eigen::DSizes<ptrdiff_t, NDIM> sizes;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    // Accumulate slices along the dimensions into the output. The number of
    // slices along dimension 'i' is simply the multiple along dimension 'i'
    // passed to the original Tile op.
    for (int i = 0; i < NDIM; ++i) {
      sizes[i] = input_dims[i] / multiples_array[i];
      indices[i] = 0;
    }

    bool first = true;
    while (true) {
      functor::TileGrad<Device, T, NDIM>()(
          context->eigen_device<Device>(), result->tensor<T, NDIM>(),
          context->input(0).tensor<T, NDIM>(), indices, sizes, first);
      first = false;
      // Increment the begin indices.
      int i = 0;
      while (i < NDIM && indices[i] / sizes[i] == multiples_array[i] - 1) {
        indices[i] = 0;
        ++i;
      }
      // We are finished if we have iterated to the maximum along all
      // dimensions.
      if (i == NDIM) {
        break;
      }
      indices[i] += sizes[i];
    }
  }

  template <typename T, int NDIM, int REDUCENDIM>
  void HandleReduce(OpKernelContext* context,
<<<<<<< HEAD
                    const std::vector<Tmultiples>& reduce_dim_in,
                    Tensor* result) {
    static_assert(NDIM >= REDUCENDIM, "Too many reduced dimensions");
    Eigen::DSizes<Eigen::DenseIndex, REDUCENDIM> reduce_dim;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> reshape_dim;
=======
                    const std::vector<int32>& reduce_dim_in, Tensor* result) {
    static_assert(NDIM >= REDUCENDIM, "Too many reduced dimensions");
    Eigen::DSizes<ptrdiff_t, REDUCENDIM> reduce_dim;
    Eigen::DSizes<ptrdiff_t, NDIM> reshape_dim;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    for (int i = 0; i < REDUCENDIM; ++i) {
      reduce_dim[i] = reduce_dim_in[i];
    }

    for (int i = 0; i < NDIM; ++i) {
      reshape_dim[i] = result->dim_size(i);
    }

    functor::ReduceAndReshape<Device, T, NDIM, REDUCENDIM>()(
        context->eigen_device<Device>(), result->tensor<T, NDIM>(),
        context->input(0).tensor<T, NDIM>(), reduce_dim, reshape_dim);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(TileGradientOp);
};

<<<<<<< HEAD
template <typename Device, typename Tmultiples>
template <DataType DT, int NDIM>
inline void TileGradientOp<Device, Tmultiples>::HandleCase(
    OpKernelContext* context, const std::vector<Tmultiples>& input_dims,
    const gtl::ArraySlice<Tmultiples>& multiples_array, Tensor* result) {
  LOG(FATAL) << "TileGradientOp: Invalid combination of Device, DT and NDIM: "
             << MakeTypeIndex<Device>().name() << ", " << DataTypeString(DT)
             << ", " << NDIM;
}

#define HANDLE_CASE(device, T, dtype, Tmultiples, ndim)                        \
  template <>                                                                  \
  template <>                                                                  \
  void TileGradientOp<device, Tmultiples>::HandleCase<dtype, ndim>(            \
      OpKernelContext * context, const std::vector<Tmultiples>& input_dims,    \
      const gtl::ArraySlice<Tmultiples>& multiples_array, Tensor* result) {    \
    HandleCaseImpl<dtype, ndim>(context, input_dims, multiples_array, result); \
  }

// 0-D handled specially above
#define HANDLE_CASE_DIM(device, T, dtype)  \
  HANDLE_CASE(device, T, dtype, int32, 1); \
  HANDLE_CASE(device, T, dtype, int32, 2); \
  HANDLE_CASE(device, T, dtype, int32, 3); \
  HANDLE_CASE(device, T, dtype, int32, 4); \
  HANDLE_CASE(device, T, dtype, int32, 5); \
  HANDLE_CASE(device, T, dtype, int32, 6); \
  HANDLE_CASE(device, T, dtype, int32, 7); \
  HANDLE_CASE(device, T, dtype, int64, 1); \
  HANDLE_CASE(device, T, dtype, int64, 2); \
  HANDLE_CASE(device, T, dtype, int64, 3); \
  HANDLE_CASE(device, T, dtype, int64, 4); \
  HANDLE_CASE(device, T, dtype, int64, 5); \
  HANDLE_CASE(device, T, dtype, int64, 6); \
  HANDLE_CASE(device, T, dtype, int64, 7);

#define HANDLE_TYPE_NAME_CPU(T) \
  HANDLE_CASE_DIM(CPUDevice, T, DataTypeToEnum<T>::value);

#define HANDLE_TYPE_NAME_GPU(T) \
  HANDLE_CASE_DIM(GPUDevice, T, DataTypeToEnum<T>::value);

TF_CALL_float(HANDLE_TYPE_NAME_CPU);
TF_CALL_double(HANDLE_TYPE_NAME_CPU);
TF_CALL_int16(HANDLE_TYPE_NAME_CPU);
TF_CALL_int32(HANDLE_TYPE_NAME_CPU);
TF_CALL_int64(HANDLE_TYPE_NAME_CPU);
TF_CALL_half(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_float(HANDLE_TYPE_NAME_GPU);
TF_CALL_double(HANDLE_TYPE_NAME_GPU);
TF_CALL_int16(HANDLE_TYPE_NAME_GPU);
TF_CALL_int32(HANDLE_TYPE_NAME_GPU);
TF_CALL_int64(HANDLE_TYPE_NAME_GPU);
TF_CALL_half(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_GPU);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if TENSORFLOW_USE_SYCL
#define HANDLE_TYPE_NAME_SYCL(T) \
  HANDLE_CASE_DIM(SYCLDevice, T, DataTypeToEnum<T>::value);

TF_CALL_float(HANDLE_TYPE_NAME_SYCL);
TF_CALL_double(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int16(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int32(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int64(HANDLE_TYPE_NAME_SYCL);
#undef HANDLE_TYPE_NAME_SYCL
#endif  // TENSORFLOW_USE_SYCL

#undef HANDLE_TYPE_NAME_CPU
#undef HANDLE_TYPE_NAME_GPU
#undef HANDLE_CASE_DIM
#undef HANDLE_CASE

REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int32>("Tmultiples"),
                        TileOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int64>("Tmultiples"),
                        TileOp<CPUDevice, int64>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int32>("Tmultiples"),
                        TileGradientOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int64>("Tmultiples"),
                        TileGradientOp<CPUDevice, int64>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_TILE(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileOp<GPUDevice, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileOp<GPUDevice, int64>);

#define REGISTER_GPU_TILE_GRAD(type)                               \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                         \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileGradientOp<GPUDevice, int32>);       \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                         \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileGradientOp<GPUDevice, int64>);

#define REGISTER_GPU(type) \
  REGISTER_GPU_TILE(type); \
  REGISTER_GPU_TILE_GRAD(type);

TF_CALL_bool(REGISTER_GPU_TILE);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
TF_CALL_half(REGISTER_GPU);
TF_CALL_int16(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU)

#undef REGISTER_GPU_TILE
#undef REGISTER_GPU_TILE_GRAD
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL(type)                                        \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_SYCL)                 \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileOp<SYCLDevice, int32>);              \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_SYCL)                 \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileOp<SYCLDevice, int64>);              \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                         \
                              .Device(DEVICE_SYCL)                 \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileGradientOp<SYCLDevice, int32>);      \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                         \
                              .Device(DEVICE_SYCL)                 \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileGradientOp<SYCLDevice, int64>);

    TF_CALL_float(REGISTER_SYCL);
TF_CALL_double(REGISTER_SYCL);

#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL

=======
template <typename Device>
template <DataType DT, int NDIM>
inline void TileGradientOp<Device>::HandleCase(
    OpKernelContext* context, const std::vector<int32>& input_dims,
    const gtl::ArraySlice<int32>& multiples_array, Tensor* result) {
  LOG(FATAL) << "TileGradientOp: Invalid combination of Device, DT and NDIM: "
             << typeid(Device).name() << ", " << DataTypeString(DT) << ", "
             << NDIM;
}

#define HANDLE_CASE(device, dtype, ndim)                                       \
  template <>                                                                  \
  template <>                                                                  \
  void TileGradientOp<device>::HandleCase<dtype, ndim>(                        \
      OpKernelContext * context, const std::vector<int32>& input_dims,         \
      const gtl::ArraySlice<int32>& multiples_array, Tensor* result) {         \
    HandleCaseImpl<dtype, ndim>(context, input_dims, multiples_array, result); \
  }

#define HANDLE_CASE_DIM_POSITIVE(device, dtype) \
  HANDLE_CASE(device, dtype, 1);                \
  HANDLE_CASE(device, dtype, 2);                \
  HANDLE_CASE(device, dtype, 3);                \
  HANDLE_CASE(device, dtype, 4);                \
  HANDLE_CASE(device, dtype, 5);

#define HANDLE_CASE_DIM(device, dtype) \
  HANDLE_CASE(device, dtype, 0);       \
  HANDLE_CASE_DIM_POSITIVE(device, dtype);

HANDLE_CASE_DIM(CPUDevice, DT_FLOAT);
HANDLE_CASE_DIM(CPUDevice, DT_DOUBLE);
HANDLE_CASE_DIM(CPUDevice, DT_INT16);
HANDLE_CASE_DIM(CPUDevice, DT_INT32);
HANDLE_CASE_DIM(CPUDevice, DT_INT64);

#if GOOGLE_CUDA
// Eigen on GPU does not handle 0-dimension data types yet.
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_FLOAT);
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_DOUBLE);
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_INT16);
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_INT32);
HANDLE_CASE_DIM_POSITIVE(GPUDevice, DT_INT64);
#endif  // GOOGLE_CUDA

#undef HANDLE_CASE_DIM_POSITIVE
#undef HANDLE_CASE_DIM
#undef HANDLE_CASE

REGISTER_KERNEL_BUILDER(Name("Tile").Device(DEVICE_CPU).HostMemory("multiples"),
                        TileOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples"),
                        TileGradientOp<CPUDevice>);

#if GOOGLE_CUDA
#define DEFINE_GPU_TYPE(T) \
  DEFINE_GPU_DIM(T, 1)     \
  DEFINE_GPU_DIM(T, 2)     \
  DEFINE_GPU_DIM(T, 3)     \
  DEFINE_GPU_DIM(T, 4)     \
  DEFINE_GPU_DIM(T, 5)

#define DEFINE_GPU_DIM(T, NDIM)                                       \
  template <>                                                         \
  void Tile<GPUDevice, T, NDIM>::operator()(                          \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor out,       \
      typename TTypes<T, NDIM>::ConstTensor in,                       \
      const Eigen::array<int32, NDIM>& broadcast_array) const;        \
  extern template struct Tile<GPUDevice, T, NDIM>;                    \
  template <>                                                         \
  void TileGrad<GPUDevice, T, NDIM>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor out,       \
      typename TTypes<T, NDIM>::ConstTensor in,                       \
      const Eigen::DSizes<ptrdiff_t, NDIM>& indices,                  \
      const Eigen::DSizes<ptrdiff_t, NDIM>& sizes, bool first) const; \
  extern template struct TileGrad<GPUDevice, T, NDIM>;                \
  template <>                                                         \
  void ReduceAndReshape<GPUDevice, T, NDIM, 1>::operator()(           \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor out,       \
      typename TTypes<T, NDIM>::ConstTensor in,                       \
      const Eigen::DSizes<ptrdiff_t, 1>& reduce_dim,                  \
      const Eigen::DSizes<ptrdiff_t, NDIM>& reshape_dim) const;       \
  extern template struct ReduceAndReshape<GPUDevice, T, NDIM, 1>;

namespace functor {
DEFINE_GPU_TYPE(float);
DEFINE_GPU_TYPE(double);
DEFINE_GPU_TYPE(int64);
DEFINE_GPU_TYPE(int32);
DEFINE_GPU_TYPE(int16);
}  // end namespace functor

#undef DEFINE_GPU_DIM
#undef DEFINE_GPU_TYPE

REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("multiples"),
                        TileOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("multiples"),
                        TileOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int16>("T")
                            .HostMemory("multiples"),
                        TileOp<GPUDevice>);

REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("multiples"),
                        TileGradientOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("multiples"),
                        TileGradientOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int16>("T")
                            .HostMemory("multiples"),
                        TileGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
