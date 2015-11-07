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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
=======
#if GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/adjust_contrast_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
<<<<<<< HEAD

// this is for v2
template struct functor::AdjustContrastv2<GPUDevice, float>;
template struct functor::AdjustContrastv2<GPUDevice, Eigen::half>;

// these are for v1
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
template struct functor::AdjustContrast<GPUDevice, uint8>;
template struct functor::AdjustContrast<GPUDevice, int8>;
template struct functor::AdjustContrast<GPUDevice, int16>;
template struct functor::AdjustContrast<GPUDevice, int32>;
template struct functor::AdjustContrast<GPUDevice, int64>;
template struct functor::AdjustContrast<GPUDevice, float>;
template struct functor::AdjustContrast<GPUDevice, double>;

}  // namespace tensorflow

<<<<<<< HEAD
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
=======
#endif  // GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
