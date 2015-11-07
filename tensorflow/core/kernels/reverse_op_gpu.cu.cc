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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
=======
#if GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/reverse_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

<<<<<<< HEAD
#define DEFINE_REVERSE(T, DIM) \
  template struct functor::Reverse<GPUDevice, T, DIM>;
#define DEFINE_REVERSE_ALL_DIMS(T) \
  DEFINE_REVERSE(T, 0)             \
  DEFINE_REVERSE(T, 1)             \
  DEFINE_REVERSE(T, 2)             \
  DEFINE_REVERSE(T, 3)             \
  DEFINE_REVERSE(T, 4)             \
  DEFINE_REVERSE(T, 5)             \
  DEFINE_REVERSE(T, 6)             \
  DEFINE_REVERSE(T, 7)             \
  DEFINE_REVERSE(T, 8)

TF_CALL_uint8(DEFINE_REVERSE_ALL_DIMS);
TF_CALL_int8(DEFINE_REVERSE_ALL_DIMS);
TF_CALL_bool(DEFINE_REVERSE_ALL_DIMS);
TF_CALL_half(DEFINE_REVERSE_ALL_DIMS);
TF_CALL_float(DEFINE_REVERSE_ALL_DIMS);
TF_CALL_double(DEFINE_REVERSE_ALL_DIMS);
TF_CALL_complex64(DEFINE_REVERSE_ALL_DIMS);
TF_CALL_complex128(DEFINE_REVERSE_ALL_DIMS);
#undef DEFINE_REVERSE
#undef DEFINE_REVERSE_ALL_DIMS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
=======
#define DEFINE_REVERSE(DIM)                                \
  template struct functor::Reverse<GPUDevice, uint8, DIM>; \
  template struct functor::Reverse<GPUDevice, int8, DIM>;  \
  template struct functor::Reverse<GPUDevice, int32, DIM>; \
  template struct functor::Reverse<GPUDevice, bool, DIM>;  \
  template struct functor::Reverse<GPUDevice, float, DIM>; \
  template struct functor::Reverse<GPUDevice, double, DIM>;
DEFINE_REVERSE(0)
DEFINE_REVERSE(1)
DEFINE_REVERSE(2)
DEFINE_REVERSE(3)
DEFINE_REVERSE(4)
DEFINE_REVERSE(5)
DEFINE_REVERSE(6)
DEFINE_REVERSE(7)
DEFINE_REVERSE(8)
#undef DEFINE_REVERSE

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
