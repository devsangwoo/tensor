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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_TYPES_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_TYPES_H_
=======
#ifndef TENSORFLOW_FRAMEWORK_TENSOR_TYPES_H_
#define TENSORFLOW_FRAMEWORK_TENSOR_TYPES_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

// Helper to define Tensor types given that the scalar is of type T.
<<<<<<< HEAD
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Tensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstTensor;

  // Unaligned Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType> >
      UnalignedTensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType> >
      UnalignedConstTensor;

  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, int>,
                           Eigen::Aligned>
      Tensor32Bit;

  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>,
      Eigen::Aligned>
      Scalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>,
                                                  Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      ConstScalar;

  // Unaligned Scalar tensor of scalar type T.
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType> >
      UnalignedScalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>,
                                                  Eigen::RowMajor, IndexType> >
      UnalignedConstScalar;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Flat;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Vec;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstVec;

  // Unaligned Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> >
      UnalignedFlat;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> >
      UnalignedConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> >
      UnalignedVec;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> >
      UnalignedConstVec;

  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Matrix;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstMatrix;

  // Unaligned Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType> >
      UnalignedMatrix;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType> >
=======
template <typename T, int NDIMS = 1>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor>,
                           Eigen::Aligned> Tensor;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor>,
                           Eigen::Aligned> ConstTensor;

  // Unaligned Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor> >
      UnalignedTensor;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor> >
      UnalignedConstTensor;

  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, int>,
                           Eigen::Aligned> Tensor32Bit;

  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor>,
      Eigen::Aligned> Scalar;
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor>,
      Eigen::Aligned> ConstScalar;

  // Unaligned Scalar tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<
      T, Eigen::Sizes<>, Eigen::RowMajor> > UnalignedScalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<
      const T, Eigen::Sizes<>, Eigen::RowMajor> > UnalignedConstScalar;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Aligned>
      Flat;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                           Eigen::Aligned> ConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Aligned>
      Vec;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                           Eigen::Aligned> ConstVec;

  // Unaligned Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor> > UnalignedFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor> >
      UnalignedConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor> > UnalignedVec;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor> >
      UnalignedConstVec;

  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>
      Matrix;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                           Eigen::Aligned> ConstMatrix;

  // Unaligned Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor> >
      UnalignedMatrix;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor> >
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      UnalignedConstMatrix;
};

typedef typename TTypes<float, 1>::Tensor32Bit::Index Index32;

template <typename DSizes>
Eigen::DSizes<Index32, DSizes::count> To32BitDims(const DSizes& in) {
  Eigen::DSizes<Index32, DSizes::count> out;
  for (int i = 0; i < DSizes::count; ++i) {
    out[i] = in[i];
  }
  return out;
}

template <typename TensorType>
typename TTypes<typename TensorType::Scalar,
                TensorType::NumIndices>::Tensor32Bit
To32Bit(TensorType in) {
  typedef typename TTypes<typename TensorType::Scalar,
                          TensorType::NumIndices>::Tensor32Bit RetType;
  return RetType(in.data(), To32BitDims(in.dimensions()));
}

}  // namespace tensorflow
<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_TYPES_H_
=======
#endif  // TENSORFLOW_FRAMEWORK_TENSOR_TYPES_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
