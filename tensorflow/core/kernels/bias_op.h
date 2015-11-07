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

#ifndef TENSORFLOW_CORE_KERNELS_BIAS_OP_H_
#define TENSORFLOW_CORE_KERNELS_BIAS_OP_H_
// Functor definition for BiasOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
=======
#ifndef TENSORFLOW_KERNELS_BIAS_OP_H_
#define TENSORFLOW_KERNELS_BIAS_OP_H_
// Functor definition for BiasOp, must be compilable by nvcc.

#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {
namespace functor {

// Functor used by BiasOp to do the computations.
template <typename Device, typename T, int Dims>
struct Bias {
  // Add "bias" to "input", broadcasting it on all dimensions but the last one.
  void operator()(const Device& d, typename TTypes<T, Dims>::ConstTensor input,
                  typename TTypes<T>::ConstVec bias,
                  typename TTypes<T, Dims>::Tensor output) {
<<<<<<< HEAD
    if (input.size() >= INT_MAX) {
      const Eigen::Index bias_size = bias.dimension(0);
      const Eigen::Index rest_size = input.size() / bias_size;
      Eigen::DSizes<Eigen::Index, 1> one_d(input.size());
      Eigen::DSizes<Eigen::Index, 1> bcast(rest_size);
      output.reshape(one_d).device(d) =
          input.reshape(one_d) + bias.broadcast(bcast);
    } else {
      const int bias_size = bias.dimension(0);
      const int rest_size = input.size() / bias_size;
      Eigen::DSizes<int, 1> one_d(input.size());
      Eigen::DSizes<int, 1> bcast(rest_size);
      To32Bit(output).reshape(one_d).device(d) =
          To32Bit(input).reshape(one_d) + To32Bit(bias).broadcast(bcast);
    }
=======
    const int bias_size = bias.dimension(0);
    const int rest_size = input.size() / bias_size;

    Eigen::DSizes<int, 2> rest_by_bias(rest_size, bias_size);
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<int, 2> rest_by_one(rest_size, 1);
    Eigen::DSizes<int, 2> one_by_bias(1, bias_size);
#else
    Eigen::IndexList<int, Eigen::type2index<1> > rest_by_one;
    rest_by_one.set(0, rest_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_bias;
    one_by_bias.set(1, bias_size);
#endif

    output.reshape(rest_by_bias).device(d) =
        input.reshape(rest_by_bias) +
        bias.reshape(one_by_bias).broadcast(rest_by_one);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
};

}  // namespace functor
}  // namespace tensorflow

<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_KERNELS_BIAS_OP_H_
=======
#endif  // TENSORFLOW_KERNELS_BIAS_OP_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
