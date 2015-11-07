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

#ifndef TENSORFLOW_CORE_KERNELS_WHERE_OP_H_
#define TENSORFLOW_CORE_KERNELS_WHERE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

#define TF_CALL_WHERE_GPU_TYPES(m) \
  TF_CALL_int8(m);                 \
  TF_CALL_uint8(m);                \
  TF_CALL_int64(m);                \
  TF_CALL_float(m);                \
  TF_CALL_double(m);               \
  TF_CALL_complex64(m);            \
  TF_CALL_complex128(m);           \
  TF_CALL_bool(m);

namespace functor {

template <typename Device, typename T, typename TIndex>
struct NumTrue {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const Device& d,
      typename TTypes<T>::ConstFlat input,
      typename TTypes<TIndex>::Scalar num_true);
};

template <typename Device, int NDIM, typename T, typename TIndex>
struct Where {
  // Copies indices of true values in input into output.  The pointer
  // found_true should sit on the host.  Compute should copy the
  // number of true elements found into it.  At the end, if
  //   *found_true != output.dimension(0),
  // then the input may have changed between the initial counting of
  // the true values and the call to Where.
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const Device& d,
      typename TTypes<T, NDIM>::ConstTensor input,
      typename TTypes<int64>::Matrix output, TIndex* found_true);
=======
#ifndef TENSORFLOW_KERNELS_WHERE_OP_H_
#define TENSORFLOW_KERNELS_WHERE_OP_H_

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

namespace functor {

template <typename Device>
struct NumTrue {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<bool>::ConstFlat input,
      TTypes<int64>::Scalar num_true) {
    num_true.device(d) = input.template cast<int64>().sum();
  }
};

template <typename Device, int NDIM>
struct Where {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<bool, NDIM>::ConstTensor input,
      typename TTypes<int64>::Matrix output) {
    Eigen::DenseIndex true_n = 0;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
    Eigen::DSizes<Eigen::DenseIndex, NDIM> strides;

    // Calculate strides for RowMajor order.
    EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                         static_cast<int>(Eigen::RowMajor)),
                        INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);

    strides[NDIM - 1] = 1;
    for (int i = NDIM - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }

    // Note, no bounds checking is done on true_n.  It is assumed that
    // the output was correctly sized via output of NumTrue::Compute.
    for (Eigen::DenseIndex n = 0; n < input.size(); ++n) {
      if (input.data()[n]) {
        WriteIndexRowMajor(output, strides, true_n, n);
        ++true_n;
      }
    }
  }

  EIGEN_ALWAYS_INLINE static void WriteIndexRowMajor(
      typename TTypes<int64>::Matrix output,
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& strides,
      Eigen::DenseIndex true_n, Eigen::DenseIndex index) {
    for (int i = 0; i < NDIM; ++i) {
      output(true_n, i) = index / strides[i];
      index %= strides[i];
    }
  }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
};

}  // namespace functor

}  // namespace tensorflow

<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_KERNELS_WHERE_OP_H_
=======
#endif  // TENSORFLOW_KERNELS_WHERE_OP_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
