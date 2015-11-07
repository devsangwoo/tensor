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
// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/cast_op.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
<<<<<<< HEAD
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/kernels/cast_op_impl.h"

=======
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/util/work_sharder.h"

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
<<<<<<< HEAD
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

#define CURRY_TYPES2(FN, arg0)   \
  FN(arg0, bool);                \
  FN(arg0, uint8);               \
  FN(arg0, uint16);              \
  FN(arg0, uint32);              \
  FN(arg0, uint64);              \
  FN(arg0, int8);                \
  FN(arg0, int16);               \
  FN(arg0, int32);               \
  FN(arg0, int64);               \
  FN(arg0, Eigen::half);         \
  FN(arg0, float);               \
  FN(arg0, double);              \
  FN(arg0, std::complex<float>); \
  FN(arg0, std::complex<double>)

CastOpBase::CastOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &external_src_dtype_));

  OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &external_dst_dtype_));

  OP_REQUIRES_OK(ctx, ctx->GetAttr("Truncate", &use_truncation_));

  // Quantized data types use the same underlying format as their non quantized
  // version so we use the non quantized implementation for casting.
  if (external_dst_dtype_ == DT_QUINT8) {
    dst_dtype_ = DT_UINT8;
  } else if (external_dst_dtype_ == DT_QINT8) {
    dst_dtype_ = DT_INT8;
  } else if (external_dst_dtype_ == DT_QINT32) {
    dst_dtype_ = DT_INT32;
  } else if (external_dst_dtype_ == DT_QINT16) {
    dst_dtype_ = DT_INT16;
  } else if (external_dst_dtype_ == DT_QUINT16) {
    dst_dtype_ = DT_UINT16;
  } else {
    dst_dtype_ = external_dst_dtype_;
  }

  if (external_src_dtype_ == DT_QUINT8) {
    src_dtype_ = DT_UINT8;
  } else if (external_src_dtype_ == DT_QINT8) {
    src_dtype_ = DT_INT8;
  } else if (external_src_dtype_ == DT_QINT32) {
    src_dtype_ = DT_INT32;
  } else if (external_src_dtype_ == DT_QINT16) {
    src_dtype_ = DT_INT16;
  } else if (external_src_dtype_ == DT_QUINT16) {
    src_dtype_ = DT_UINT16;
  } else {
    src_dtype_ = external_src_dtype_;
  }
}

void CastOpBase::Compute(OpKernelContext* ctx) {
  const Tensor& inp = ctx->input(0);
  if (work_ == nullptr) {
    ctx->set_output(0, inp);
  } else {
    Tensor in;
    if (external_src_dtype_ != src_dtype_) {
      // If the type is a quantized type we need to do a bitcast since the
      // src_dtype_ is different from external_src_type_.
      OP_REQUIRES_OK(ctx, in.BitcastFrom(inp, src_dtype_, inp.shape()));
    } else {
      in = inp;
    }
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &out));
    out->set_dtype(dst_dtype_);
    work_(ctx, in, out, use_truncation_);
    out->set_dtype(external_dst_dtype_);
  }
}

Status CastOpBase::Unimplemented() {
  return errors::Unimplemented("Cast ", DataTypeString(external_src_dtype_),
                               " to ", DataTypeString(external_dst_dtype_),
                               " is not supported");
}

CpuCastOp::CpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
  OP_REQUIRES_OK(ctx, Prepare());
}

Status CpuCastOp::Prepare() {
  if (external_src_dtype_ == external_dst_dtype_) {
    work_ = nullptr;  // Identity
    return Status::OK();
  }
  if (src_dtype_ == DT_BOOL) {
    work_ = GetCpuCastFromBool(dst_dtype_);
  } else if (src_dtype_ == DT_UINT8) {
    work_ = GetCpuCastFromUint8(dst_dtype_);
  } else if (src_dtype_ == DT_UINT16) {
    work_ = GetCpuCastFromUint16(dst_dtype_);
  } else if (src_dtype_ == DT_UINT32) {
    work_ = GetCpuCastFromUint32(dst_dtype_);
  } else if (src_dtype_ == DT_UINT64) {
    work_ = GetCpuCastFromUint64(dst_dtype_);
  } else if (src_dtype_ == DT_INT8) {
    work_ = GetCpuCastFromInt8(dst_dtype_);
  } else if (src_dtype_ == DT_INT16) {
    work_ = GetCpuCastFromInt16(dst_dtype_);
  } else if (src_dtype_ == DT_INT32) {
    work_ = GetCpuCastFromInt32(dst_dtype_);
  } else if (src_dtype_ == DT_INT64) {
    work_ = GetCpuCastFromInt64(dst_dtype_);
  } else if (src_dtype_ == DT_HALF) {
    work_ = GetCpuCastFromHalf(dst_dtype_);
  } else if (src_dtype_ == DT_FLOAT) {
    work_ = GetCpuCastFromFloat(dst_dtype_);
  } else if (src_dtype_ == DT_DOUBLE) {
    work_ = GetCpuCastFromDouble(dst_dtype_);
  } else if (src_dtype_ == DT_COMPLEX64) {
    work_ = GetCpuCastFromComplex64(dst_dtype_);
  } else if (src_dtype_ == DT_COMPLEX128) {
    work_ = GetCpuCastFromComplex128(dst_dtype_);
  } else if (src_dtype_ == DT_BFLOAT16) {
    work_ = GetCpuCastFromBfloat(dst_dtype_);
  }

  // TODO(sesse): If CPU casting to or from Eigen::half ever becomes a
  // bottleneck, we could probably implement specialized support for
  // vectorized versions (not the least based on F16C for Haswell
  // or newer).

  return work_ == nullptr ? Unimplemented() : Status::OK();
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
class GpuCastOp : public CastOpBase {
 public:
  explicit GpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 private:
  Status Prepare() {
    if (external_src_dtype_ == external_dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    if (src_dtype_ == DT_BOOL) {
      work_ = GetGpuCastFromBool(dst_dtype_);
    } else if (src_dtype_ == DT_UINT8) {
      work_ = GetGpuCastFromUint8(dst_dtype_);
    } else if (src_dtype_ == DT_UINT16) {
      work_ = GetGpuCastFromUint16(dst_dtype_);
    } else if (src_dtype_ == DT_UINT32) {
      work_ = GetGpuCastFromUint32(dst_dtype_);
    } else if (src_dtype_ == DT_UINT64) {
      work_ = GetGpuCastFromUint64(dst_dtype_);
    } else if (src_dtype_ == DT_INT8) {
      work_ = GetGpuCastFromInt8(dst_dtype_);
    } else if (src_dtype_ == DT_INT16) {
      work_ = GetGpuCastFromInt16(dst_dtype_);
    } else if (src_dtype_ == DT_INT32) {
      work_ = GetGpuCastFromInt32(dst_dtype_);
    } else if (src_dtype_ == DT_INT64) {
      work_ = GetGpuCastFromInt64(dst_dtype_);
    } else if (src_dtype_ == DT_HALF) {
      work_ = GetGpuCastFromHalf(dst_dtype_);
    } else if (src_dtype_ == DT_FLOAT) {
      work_ = GetGpuCastFromFloat(dst_dtype_);
    } else if (src_dtype_ == DT_DOUBLE) {
      work_ = GetGpuCastFromDouble(dst_dtype_);
    } else if (src_dtype_ == DT_COMPLEX64) {
      work_ = GetGpuCastFromComplex64(dst_dtype_);
    } else if (src_dtype_ == DT_COMPLEX128) {
      work_ = GetGpuCastFromComplex128(dst_dtype_);
    } else if (src_dtype_ == DT_BFLOAT16) {
      work_ = GetGpuCastFromBfloat(dst_dtype_);
    }

    return work_ == nullptr ? Unimplemented() : Status::OK();
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef CAST_CASE

REGISTER_KERNEL_BUILDER(Name("Cast").Device(DEVICE_CPU), CpuCastOp);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define REGISTER_CAST_GPU(srctype, dsttype)                    \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_GPU),             \
                          GpuCastOp)

CURRY_TYPES2(REGISTER_CAST_GPU, bool);
CURRY_TYPES2(REGISTER_CAST_GPU, uint8);
CURRY_TYPES2(REGISTER_CAST_GPU, uint16);
CURRY_TYPES2(REGISTER_CAST_GPU, uint32);
CURRY_TYPES2(REGISTER_CAST_GPU, uint64);
CURRY_TYPES2(REGISTER_CAST_GPU, int8);
CURRY_TYPES2(REGISTER_CAST_GPU, int16);
CURRY_TYPES2(REGISTER_CAST_GPU, int32);
CURRY_TYPES2(REGISTER_CAST_GPU, int64);
CURRY_TYPES2(REGISTER_CAST_GPU, Eigen::half);
CURRY_TYPES2(REGISTER_CAST_GPU, float);
CURRY_TYPES2(REGISTER_CAST_GPU, double);
CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<float>);
CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<double>);
REGISTER_CAST_GPU(float, bfloat16);
REGISTER_CAST_GPU(bfloat16, float);

#undef REGISTER_CAST_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
class SyclCastOp : public CastOpBase {
 public:
  explicit SyclCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 private:
  Status Prepare() {
    if (external_src_dtype_ == external_dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    if (src_dtype_ == DT_BOOL) {
      work_ = GetSyclCastFromBool(dst_dtype_);
    } else if (src_dtype_ == DT_INT32) {
      work_ = GetSyclCastFromInt32(dst_dtype_);
    } else if (src_dtype_ == DT_INT64) {
      work_ = GetSyclCastFromInt64(dst_dtype_);
    } else if (src_dtype_ == DT_FLOAT) {
      work_ = GetSyclCastFromFloat(dst_dtype_);
    } else if (src_dtype_ == DT_DOUBLE) {
      work_ = GetSyclCastFromDouble(dst_dtype_);
    }

    return work_ == nullptr ? Unimplemented() : Status::OK();
  }
};

#define REGISTER_CAST_SYCL(srctype, dsttype)                   \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_SYCL),            \
                          SyclCastOp)
CURRY_TYPES2(REGISTER_CAST_SYCL, bool);
CURRY_TYPES2(REGISTER_CAST_SYCL, int32);
CURRY_TYPES2(REGISTER_CAST_SYCL, int64);
CURRY_TYPES2(REGISTER_CAST_SYCL, float);
CURRY_TYPES2(REGISTER_CAST_SYCL, double);

#undef REGISTER_CAST_SYCL

#endif  // TENSORFLOW_USE_SYCL

#undef CURRY_TYPES2
=======

namespace functor {

template <typename Device, typename Tout, typename Tin>
void CastMaybeInline(const Device& d, typename TTypes<Tout>::Flat o,
                     typename TTypes<Tin>::ConstFlat i) {
  if (o.size() * (sizeof(Tin) + sizeof(Tout)) < 131072) {
    // Small cast on a CPU: do inline
    o = i.template cast<Tout>();
  } else {
    o.device(d) = i.template cast<Tout>();
  }
}

template <typename O, typename I>
struct CastFunctor<CPUDevice, O, I> {
  void operator()(const CPUDevice& d, typename TTypes<O>::Flat o,
                  typename TTypes<I>::ConstFlat i) {
    CastMaybeInline<CPUDevice, O, I>(d, o, i);
  }
};

}  // namespace functor

#define CAST_CASE(DEVICE, IN, OUT)                                         \
  if (DataTypeToEnum<IN>::value == src_dtype_ &&                           \
      DataTypeToEnum<OUT>::value == dst_dtype_) {                          \
    work_ = [](OpKernelContext* ctx, const Tensor& inp, Tensor* out) {     \
      functor::CastFunctor<DEVICE, OUT, IN> func;                          \
      func(ctx->eigen_device<DEVICE>(), out->flat<OUT>(), inp.flat<IN>()); \
    };                                                                     \
    return Status::OK();                                                   \
  }

class CastOpBase : public OpKernel {
 public:
  explicit CastOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &dst_dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    if (work_ == nullptr) {
      ctx->set_output(0, inp);
    } else {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &out));
      work_(ctx, inp, out);
    }
  }

 protected:
  DataType src_dtype_;
  DataType dst_dtype_;
  std::function<void(OpKernelContext*, const Tensor&, Tensor*)> work_ = nullptr;

  virtual Status Prepare() = 0;
  Status Unimplemented() {
    return errors::Unimplemented("Cast ", DataTypeString(src_dtype_), " to ",
                                 DataTypeString(dst_dtype_),
                                 " is not supported");
  }

  TF_DISALLOW_COPY_AND_ASSIGN(CastOpBase);
};

class CpuCastOp : public CastOpBase {
 public:
  explicit CpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 protected:
  Status Prepare() override {
    if (src_dtype_ == dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    CAST_CASE(CPUDevice, bool, float);
    CAST_CASE(CPUDevice, bool, int32);
    CAST_CASE(CPUDevice, bool, double);
    CAST_CASE(CPUDevice, double, float);
    CAST_CASE(CPUDevice, double, int32);
    CAST_CASE(CPUDevice, double, int64);
    CAST_CASE(CPUDevice, float, double);
    CAST_CASE(CPUDevice, float, uint8);
    CAST_CASE(CPUDevice, float, int32);
    CAST_CASE(CPUDevice, float, int64);
    CAST_CASE(CPUDevice, int32, double);
    CAST_CASE(CPUDevice, int32, float);
    CAST_CASE(CPUDevice, int32, uint8);
    CAST_CASE(CPUDevice, int32, int64);
    CAST_CASE(CPUDevice, int64, double);
    CAST_CASE(CPUDevice, int64, float);
    CAST_CASE(CPUDevice, int64, int32);
    CAST_CASE(CPUDevice, uint8, float);
    CAST_CASE(CPUDevice, uint8, int32);
    CAST_CASE(CPUDevice, uint8, int64);
    CAST_CASE(CPUDevice, uint8, double);
    if (src_dtype_ == DT_BFLOAT16 && dst_dtype_ == DT_FLOAT) {
      work_ = [](OpKernelContext* ctx, const Tensor& inp, Tensor* out) {
        int64 N = out->NumElements();
        auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
        int num_threads =
            std::min<int>(std::min(4, worker_threads->num_threads), N / 4096);
        if (num_threads < 1) {
          BFloat16ToFloat(inp.flat<bfloat16>().data(),
                          out->flat<float>().data(), N);
        } else {
          auto work = [&inp, &out](int64 start, int64 end) {
            BFloat16ToFloat(inp.flat<bfloat16>().data() + start,
                            out->flat<float>().data() + start, end - start);
          };
          Shard(num_threads, worker_threads->workers, N, 100, work);
        }
      };
      return Status::OK();
    }
    if (src_dtype_ == DT_FLOAT && dst_dtype_ == DT_BFLOAT16) {
      work_ = [](OpKernelContext* ctx, const Tensor& inp, Tensor* out) {
        int64 N = out->NumElements();
        auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
        int num_threads =
            std::min<int>(std::min(4, worker_threads->num_threads), N / 4096);
        if (num_threads < 1) {
          FloatToBFloat16(inp.flat<float>().data(),
                          out->flat<bfloat16>().data(), N);
        } else {
          auto work = [&inp, &out](int64 start, int64 end) {
            FloatToBFloat16(inp.flat<float>().data() + start,
                            out->flat<bfloat16>().data() + start, end - start);
          };
          Shard(num_threads, worker_threads->workers, N, 100, work);
        }
      };
      return Status::OK();
    }
    return Unimplemented();
  }
};

class GpuCastOp : public CastOpBase {
 public:
  explicit GpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 protected:
  Status Prepare() override {
    if (src_dtype_ == dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    CAST_CASE(GPUDevice, bfloat16, float);
    CAST_CASE(GPUDevice, bool, float);
    CAST_CASE(GPUDevice, double, float);
    CAST_CASE(GPUDevice, double, int64);
    CAST_CASE(GPUDevice, float, bfloat16);
    CAST_CASE(GPUDevice, float, double);
    CAST_CASE(GPUDevice, float, int64);
    CAST_CASE(GPUDevice, int64, double);
    CAST_CASE(GPUDevice, int64, float);
    CAST_CASE(GPUDevice, uint8, float);
    CAST_CASE(GPUDevice, float, uint8);
    CAST_CASE(GPUDevice, bool, int32);
    CAST_CASE(GPUDevice, double, int32);
    CAST_CASE(GPUDevice, float, int32);
    CAST_CASE(GPUDevice, int32, double);
    CAST_CASE(GPUDevice, int32, float);
    CAST_CASE(GPUDevice, int32, int64);
    CAST_CASE(GPUDevice, int64, int32);
    return Unimplemented();
  }
};

#undef CAST_CASE

REGISTER_KERNEL_BUILDER(Name("Cast").Device(DEVICE_CPU), CpuCastOp);

#if GOOGLE_CUDA
#define REGISTER_CAST_GPU(srctype, dsttype)                    \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_GPU),             \
                          GpuCastOp);
REGISTER_CAST_GPU(bfloat16, float);
REGISTER_CAST_GPU(bool, float);
REGISTER_CAST_GPU(double, float);
REGISTER_CAST_GPU(double, int64);
REGISTER_CAST_GPU(float, bfloat16);
REGISTER_CAST_GPU(float, double);
REGISTER_CAST_GPU(float, int64);
REGISTER_CAST_GPU(int64, double);
REGISTER_CAST_GPU(int64, float);
REGISTER_CAST_GPU(uint8, float);
REGISTER_CAST_GPU(float, uint8);
REGISTER_CAST_GPU(bool, int32);
REGISTER_CAST_GPU(double, int32);
REGISTER_CAST_GPU(float, int32);
REGISTER_CAST_GPU(int32, double);
REGISTER_CAST_GPU(int32, float);
REGISTER_CAST_GPU(int32, int64);
REGISTER_CAST_GPU(int64, int32);
#undef REGISTER_CAST_GPU
#endif  // GOOGLE_CUDA
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// HostCast differs from Cast in that its input and output are in host memory.
REGISTER_KERNEL_BUILDER(Name("_HostCast").Device(DEVICE_CPU), CpuCastOp);
REGISTER_KERNEL_BUILDER(
<<<<<<< HEAD
    Name("_HostCast").Device(DEVICE_DEFAULT).HostMemory("x").HostMemory("y"),
    CpuCastOp);
=======
    Name("_HostCast").Device(DEVICE_GPU).HostMemory("x").HostMemory("y"),
    CpuCastOp);

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // end namespace tensorflow
