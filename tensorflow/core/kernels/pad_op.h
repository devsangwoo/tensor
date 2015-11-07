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

#ifndef TENSORFLOW_CORE_KERNELS_PAD_OP_H_
#define TENSORFLOW_CORE_KERNELS_PAD_OP_H_
// Functor definition for PadOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
=======
#ifndef TENSORFLOW_KERNELS_PAD_OP_H_
#define TENSORFLOW_KERNELS_PAD_OP_H_
// Functor definition for PadOp, must be compilable by nvcc.

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {
namespace functor {

// Functor used by PadOp to do the computations.
<<<<<<< HEAD
template <typename Device, typename T, typename Tpadding, int Dims>
struct Pad {
  // Pad "input" into "output", as specified by "paddings" and "pad_value".
  // See pad_op.cc for details.
  void operator()(const Device& d, typename TTypes<T, Dims>::Tensor output,
                  typename TTypes<T, Dims>::ConstTensor input,
                  Eigen::array<Eigen::IndexPair<Tpadding>, Dims> paddings,
                  T pad_value) {
    if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value &&
        (output.size() <= std::numeric_limits<int32>::max())) {
      To32Bit(output).device(d) = To32Bit(input).pad(paddings, pad_value);
    } else {
      output.device(d) = input.pad(paddings, pad_value);
    }
  }
};

template <typename Device, typename T, typename Tpadding>
struct Pad<Device, T, Tpadding, 0> {
  // In the scalar case we simply copy the input.
  void operator()(const Device& d, typename TTypes<T, 0>::Tensor output,
                  typename TTypes<T, 0>::ConstTensor input,
                  Eigen::array<Eigen::IndexPair<Tpadding>, 0>, T) {
    output.device(d) = input;
  }
};
}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_PAD_OP_H_
=======
template <typename Device, typename T, int Dims>
struct Pad {
  // Pad "input" into "output", as specified by "paddings".  See pad_op.cc for
  // details.
  void operator()(const Device& d, typename TTypes<T, Dims>::Tensor output,
                  typename TTypes<T, Dims>::ConstTensor input,
                  Eigen::array<std::pair<int32, int32>, Dims> paddings) {
    output.device(d) = input.pad(paddings);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_PAD_OP_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
