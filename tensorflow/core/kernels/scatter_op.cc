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
// See docs in ../ops/state_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

// Check whether updates.shape = indices.shape + params.shape[1:]
static bool ValidShapes(const Tensor& params, const Tensor& updates,
                        const Tensor& indices) {
  if (updates.dims() == 0) return true;
  if (updates.dims() != indices.dims() + params.dims() - 1) return false;
  for (int d = 0; d < indices.dims(); d++) {
    if (updates.dim_size(d) != indices.dim_size(d)) {
      return false;
    }
  }
  for (int d = 1; d < params.dims(); d++) {
    if (params.dim_size(d) != updates.dim_size(d - 1 + indices.dims())) {
      return false;
    }
  }
  return true;
}

static void DoValidationChecking(OpKernelContext* c, const Tensor& params,
                                 const Tensor& indices, const Tensor& updates) {
  OP_REQUIRES(c, params.IsInitialized(),
              errors::FailedPrecondition("Null ref for params"));
  OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
              errors::InvalidArgument("params must be at least 1-D, got shape ",
                                      params.shape().DebugString()));
  OP_REQUIRES(
      c, ValidShapes(params, updates, indices),
      errors::InvalidArgument("Must have updates.shape = indices.shape + "
                              "params.shape[1:] or updates.shape = [], got ",
                              "updates.shape ", updates.shape().DebugString(),
                              ", indices.shape ", indices.shape().DebugString(),
                              ", params.shape ", params.shape().DebugString()));
}

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
=======
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

enum class UpdateOp { ASSIGN, ADD, SUB };

template <class T, typename Index, UpdateOp op>
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
class ScatterUpdateOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here.  Should we have the framework do some sort of
  //   integer promotion automatically, or should that be something
  //   that users have to do explicitly with a conversion operator
  //   in the graph?
  explicit ScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* c) override {
    if (use_exclusive_lock_) {
      // Hold mutex while we apply updates
      mutex_lock l(*c->input_ref_mutex(0));
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  bool use_exclusive_lock_;

<<<<<<< HEAD
  void DoCompute(OpKernelContext* c) {
    Tensor params = c->mutable_input(0, use_exclusive_lock_);
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    DoValidationChecking(c, params, indices, updates);
    if (!c->status().ok()) return;

    // Check that we have enough index space
    const int64 N_big = indices.NumElements();
    OP_REQUIRES(
        c, N_big <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("indices has too many elements for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", N_big, " > ",
                                std::numeric_limits<Index>::max()));
    const Index N = static_cast<Index>(indices.NumElements());
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    // We always return the input ref.
    c->forward_ref_input_to_ref_output(0, 0);

    if (N > 0) {
      auto indices_flat = indices.flat<Index>();
      auto params_flat = params.flat_outer_dims<T>();

      if (TensorShapeUtils::IsScalar(updates.shape()) ||
          IsLegacyScalar(updates.shape())) {
        const auto update = updates.scalar<T>();
        functor::ScatterScalarFunctor<Device, T, Index, op> functor;
        const Index bad_i = functor(c, c->template eigen_device<Device>(),
                                    params_flat, update, indices_flat);
        OP_REQUIRES(c, bad_i < 0,
                    errors::InvalidArgument(
                        "indices", SliceDebugString(indices.shape(), bad_i),
                        " = ", indices_flat(bad_i), " is not in [0, ",
                        params.dim_size(0), ")"));
      } else {
        auto updates_flat =
            updates.shaped<T, 2>({N, updates.NumElements() / N});

        functor::ScatterFunctor<Device, T, Index, op> functor;
        const Index bad_i = functor(c, c->template eigen_device<Device>(),
                                    params_flat, updates_flat, indices_flat);
        OP_REQUIRES(c, bad_i < 0,
                    errors::InvalidArgument(
                        "indices", SliceDebugString(indices.shape(), bad_i),
                        " = ", indices_flat(bad_i), " is not in [0, ",
                        params.dim_size(0), ")"));
      }
    }
  }
};

#ifdef TENSORFLOW_USE_SYCL
template <typename T, typename Index, scatter_op::UpdateOp op>
class ScatterUpdateOp<SYCLDevice, T, Index, op> : public OpKernel {
 public:
  explicit ScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* c) override {
    if (use_exclusive_lock_) {
      // Hold mutex while we apply updates
      mutex_lock l(*c->input_ref_mutex(0));
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  bool use_exclusive_lock_;

  void DoCompute(OpKernelContext* c) {
    Tensor params = c->mutable_input(0, use_exclusive_lock_);
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    DoValidationChecking(c, params, indices, updates);
    if (!c->status().ok()) return;

    // Check that we have enough index space
    const int64 N_big = indices.NumElements();
    OP_REQUIRES(
        c, N_big <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("indices has too many elements for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", N_big, " > ",
                                std::numeric_limits<Index>::max()));
    const Index N = static_cast<Index>(indices.NumElements());
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));
=======
  // Check whether updates.shape = indices.shape + params.shape[1:]
  static bool ValidShapes(const Tensor& params, const Tensor& updates,
                          const Tensor& indices) {
    if (updates.dims() != indices.dims() + params.dims() - 1) return false;
    for (int d = 0; d < indices.dims(); d++) {
      if (updates.dim_size(d) != indices.dim_size(d)) {
        return false;
      }
    }
    for (int d = 1; d < params.dims(); d++) {
      if (params.dim_size(d) != updates.dim_size(d - 1 + indices.dims())) {
        return false;
      }
    }
    return true;
  }

  void DoCompute(OpKernelContext* c) {
    Tensor Tparams = c->mutable_input(0, use_exclusive_lock_);
    OP_REQUIRES(c, Tparams.IsInitialized(),
                errors::FailedPrecondition("Null ref for params"));
    const Tensor& Tindices = c->input(1);
    const Tensor& Tupdates = c->input(2);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(Tparams.shape()),
        errors::InvalidArgument("params must be at least 1-D, got shape ",
                                Tparams.shape().ShortDebugString()));
    OP_REQUIRES(
        c, ValidShapes(Tparams, Tupdates, Tindices),
        errors::InvalidArgument(
            "Must have updates.shape = indices.shape + params.shape[1:], got ",
            "updates.shape ", Tupdates.shape().ShortDebugString(),
            ", indices.shape ", Tindices.shape().ShortDebugString(),
            ", params.shape ", Tparams.shape().ShortDebugString()));
    const Index N = Tindices.NumElements();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    // We always return the input ref.
    c->forward_ref_input_to_ref_output(0, 0);

    if (N > 0) {
<<<<<<< HEAD
      auto index_size = indices.NumElements() * sizeof(Index);
      Tensor indices_host = Tensor(indices.dtype(), indices.shape());

      auto src_ptr = GetBase(&indices);
      auto dst_ptr = GetBase(&indices_host);

      c->eigen_sycl_device().memcpyDeviceToHost(
          dst_ptr, static_cast<const Index*>(src_ptr), index_size);

      auto indices_flat = indices_host.flat<Index>();
      auto params_flat = params.flat_outer_dims<T>();

      if (TensorShapeUtils::IsScalar(updates.shape())) {
        const auto update = updates.scalar<T>();

        functor::ScatterScalarFunctorSYCL<T, Index, op> functor;
        const Index bad_i = functor(c, c->template eigen_device<SYCLDevice>(),
                                    params_flat, update, indices_flat);
        OP_REQUIRES(c, bad_i < 0,
                    errors::InvalidArgument(
                        "indices", SliceDebugString(indices.shape(), bad_i),
                        " = ", indices_flat(bad_i), " is not in [0, ",
                        params.dim_size(0), ")"));
      } else {
        auto updates_flat =
            updates.shaped<T, 2>({N, updates.NumElements() / N});

        functor::ScatterFunctorSYCL<T, Index, op> functor;
        const Index bad_i = functor(c, c->template eigen_device<SYCLDevice>(),
                                    params_flat, updates_flat, indices_flat);
        OP_REQUIRES(c, bad_i < 0,
                    errors::InvalidArgument(
                        "indices", SliceDebugString(indices.shape(), bad_i),
                        " = ", indices_flat(bad_i), " is not in [0, ",
                        params.dim_size(0), ")"));
=======
      const Index first_dim_size = Tparams.dim_size(0);
      // Validate all the indices are in range
      auto Tindices_vec = Tindices.flat<Index>();
      for (Index i = 0; i < N; i++) {
        const Index index = Tindices_vec(i);
        OP_REQUIRES(c, index >= 0 && index < first_dim_size,
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
      }
      auto Tparams_flat = Tparams.flat_outer_dims<T>();
      auto Tupdates_flat =
          Tupdates.shaped<T, 2>({N, Tupdates.NumElements() / N});
      for (Index i = 0; i < N; i++) {
        // Copy last Ndim-1 dimensions of Tupdates[i] to
        // Tparams[Tindices[i]]
        switch (op) {
          case UpdateOp::ASSIGN: {
            Tparams_flat.template chip<0>(Tindices_vec(i)) =
                Tupdates_flat.template chip<0>(i);
            break;
          }
          case UpdateOp::ADD: {
            Tparams_flat.template chip<0>(Tindices_vec(i)) +=
                Tupdates_flat.template chip<0>(i);
            break;
          }
          case UpdateOp::SUB: {
            Tparams_flat.template chip<0>(Tindices_vec(i)) -=
                Tupdates_flat.template chip<0>(i);
            break;
          }
        }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      }
    }
  }
};
<<<<<<< HEAD
#endif  // TENSORFLOW_USE_SYCL

#define REGISTER_SCATTER_KERNEL_INDEX(type, index_type, dev, name, op) \
  REGISTER_KERNEL_BUILDER(Name(name)                                   \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          ScatterUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_KERNEL_INDEX(type, int64, dev, name, op);

#define REGISTER_SCATTER_ARITHMETIC(type, dev)                                 \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterAdd", scatter_op::UpdateOp::ADD); \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterDiv", scatter_op::UpdateOp::DIV); \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterMul", scatter_op::UpdateOp::MUL); \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterSub", scatter_op::UpdateOp::SUB);

#define REGISTER_SCATTER_MINMAX(type, dev)                                     \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterMin", scatter_op::UpdateOp::MIN); \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterMax", scatter_op::UpdateOp::MAX);

#define REGISTER_SCATTER_UPDATE(type, dev)            \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterUpdate", \
                          scatter_op::UpdateOp::ASSIGN);

// Registers CPU kernels.
#define REGISTER_SCATTER_ARITHMETIC_CPU(type) \
  REGISTER_SCATTER_ARITHMETIC(type, CPU);

#define REGISTER_SCATTER_MINMAX_CPU(type) REGISTER_SCATTER_MINMAX(type, CPU);

#define REGISTER_SCATTER_UPDATE_CPU(type) REGISTER_SCATTER_UPDATE(type, CPU);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_SCATTER_MINMAX_CPU);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ARITHMETIC_CPU);
TF_CALL_ALL_TYPES(REGISTER_SCATTER_UPDATE_CPU);

// Registers GPU kernels.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_SCATTER_ARITHMETIC_GPU(type) \
  REGISTER_SCATTER_ARITHMETIC(type, GPU);

#define REGISTER_SCATTER_MINMAX_GPU(type) REGISTER_SCATTER_MINMAX(type, GPU);

#define REGISTER_SCATTER_UPDATE_GPU(type) REGISTER_SCATTER_UPDATE(type, GPU);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ARITHMETIC_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_MINMAX_GPU);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_UPDATE_GPU);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Registers GPU kernels.
#if TENSORFLOW_USE_SYCL
#define REGISTER_SCATTER_ARITHMETIC_SYCL(type) \
  REGISTER_SCATTER_ARITHMETIC(type, SYCL);

#define REGISTER_SCATTER_MINMAX_SYCL(type) REGISTER_SCATTER_MINMAX(type, SYCL);

#define REGISTER_SCATTER_UPDATE_SYCL(type) REGISTER_SCATTER_UPDATE(type, SYCL);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ARITHMETIC_SYCL);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_MINMAX_SYCL);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_UPDATE_SYCL);

#undef REGISTER_SCATTER_ARITHMETIC_SYCL
#undef REGISTER_SCATTER_MINMAX_SYCL
#undef REGISTER_SCATTER_UPDATE_SYCL
#endif  // TENSORFLOW_USE_SYCL

#undef REGISTER_SCATTER_ARITHMETIC
#undef REGISTER_SCATTER_ARITHMETIC_CPU
#undef REGISTER_SCATTER_ARITHMETIC_GPU
#undef REGISTER_SCATTER_MINMAX
#undef REGISTER_SCATTER_MINMAX_CPU
#undef REGISTER_SCATTER_MINMAX_GPU
#undef REGISTER_SCATTER_UPDATE
#undef REGISTER_SCATTER_UPDATE_CPU
#undef REGISTER_SCATTER_UPDATE_GPU
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX
=======

#define REGISTER_SCATTER_UPDATE(type, index_type)  \
  REGISTER_KERNEL_BUILDER(                         \
      Name("ScatterUpdate")                        \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<type>("T")               \
          .TypeConstraint<index_type>("Tindices"), \
      ScatterUpdateOp<type, index_type, UpdateOp::ASSIGN>);

#define REGISTER_SCATTER_UPDATE_INT32(type) REGISTER_SCATTER_UPDATE(type, int32)
#define REGISTER_SCATTER_UPDATE_INT64(type) REGISTER_SCATTER_UPDATE(type, int64)

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_UPDATE_INT32);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_UPDATE_INT64);

#undef REGISTER_SCATTER_UPDATE_INT64
#undef REGISTER_SCATTER_UPDATE_INT32
#undef REGISTER_SCATTER_UPDATE

#define REGISTER_SCATTER_ADD(type, index_type)                         \
  REGISTER_KERNEL_BUILDER(Name("ScatterAdd")                           \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          ScatterUpdateOp<type, index_type, UpdateOp::ADD>);

#define REGISTER_SCATTER_ADD_INT32(type) REGISTER_SCATTER_ADD(type, int32)
#define REGISTER_SCATTER_ADD_INT64(type) REGISTER_SCATTER_ADD(type, int64)

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ADD_INT32);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ADD_INT64);

#undef REGISTER_SCATTER_ADD_INT32
#undef REGISTER_SCATTER_ADD_INT64
#undef REGISTER_SCATTER_ADD

#define REGISTER_SCATTER_SUB(type, index_type)                         \
  REGISTER_KERNEL_BUILDER(Name("ScatterSub")                           \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          ScatterUpdateOp<type, index_type, UpdateOp::SUB>);

#define REGISTER_SCATTER_SUB_INT32(type) REGISTER_SCATTER_SUB(type, int32)
#define REGISTER_SCATTER_SUB_INT64(type) REGISTER_SCATTER_SUB(type, int64)

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_SUB_INT32);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_SUB_INT64);

#undef REGISTER_SCATTER_SUB_INT64
#undef REGISTER_SCATTER_SUB_INT32
#undef REGISTER_SCATTER_SUB
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace tensorflow
