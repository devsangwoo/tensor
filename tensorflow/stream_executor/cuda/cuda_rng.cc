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

#include "tensorflow/stream_executor/cuda/cuda_rng.h"

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
=======
#include "tensorflow/stream_executor/cuda/cuda_rng.h"

#include <dlfcn.h>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/dso_loader.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/rng.h"
<<<<<<< HEAD
// clang-format off
#include "third_party/gpus/cuda/include/curand.h"
// clang-format on
=======
#include "third_party/gpus/cuda/include/curand.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// Formats curandStatus_t to output prettified values into a log stream.
std::ostream &operator<<(std::ostream &in, const curandStatus_t &status) {
#define OSTREAM_CURAND_STATUS(__name) \
  case CURAND_STATUS_##__name:        \
    in << "CURAND_STATUS_" #__name;   \
    return in;

  switch (status) {
    OSTREAM_CURAND_STATUS(SUCCESS)
    OSTREAM_CURAND_STATUS(VERSION_MISMATCH)
    OSTREAM_CURAND_STATUS(NOT_INITIALIZED)
    OSTREAM_CURAND_STATUS(ALLOCATION_FAILED)
    OSTREAM_CURAND_STATUS(TYPE_ERROR)
    OSTREAM_CURAND_STATUS(OUT_OF_RANGE)
    OSTREAM_CURAND_STATUS(LENGTH_NOT_MULTIPLE)
    OSTREAM_CURAND_STATUS(LAUNCH_FAILURE)
    OSTREAM_CURAND_STATUS(PREEXISTING_FAILURE)
    OSTREAM_CURAND_STATUS(INITIALIZATION_FAILED)
    OSTREAM_CURAND_STATUS(ARCH_MISMATCH)
    OSTREAM_CURAND_STATUS(INTERNAL_ERROR)
    default:
      in << "curandStatus_t(" << static_cast<int>(status) << ")";
      return in;
  }
}

<<<<<<< HEAD
namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kGpuRandPlugin);

GpuRng::GpuRng(GpuExecutor* parent) : parent_(parent), rng_(nullptr) {}

GpuRng::~GpuRng() {
  if (rng_ != nullptr) {
    cuda::ScopedActivateExecutorContext sac(parent_);
    curandDestroyGenerator(rng_);
  }
}

bool GpuRng::Init() {
  absl::MutexLock lock(&mu_);
  CHECK(rng_ == nullptr);

  cuda::ScopedActivateExecutorContext sac(parent_);
  curandStatus_t ret = curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT);
=======
namespace perftools {
namespace gputools {
namespace cuda {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuRandPlugin);

namespace dynload {

#define PERFTOOLS_GPUTOOLS_CURAND_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                            \
    static const char *kName;                                               \
    using FuncPointerT = std::add_pointer<decltype(::__name)>::type;        \
    static void *GetDsoHandle() {                                           \
      static auto status = internal::CachedDsoLoader::GetCurandDsoHandle(); \
      return status.ValueOrDie();                                           \
    }                                                                       \
    static FuncPointerT DynLoad() {                                         \
      static void *f = dlsym(GetDsoHandle(), kName);                        \
      CHECK(f != nullptr) << "could not find " << kName                     \
                          << " in curand DSO; dlerror: " << dlerror();      \
      return reinterpret_cast<FuncPointerT>(f);                             \
    }                                                                       \
    template <typename... Args>                                             \
    curandStatus_t operator()(CUDAExecutor * parent, Args... args) {        \
      cuda::ScopedActivateExecutorContext sac{parent};                      \
      return DynLoad()(args...);                                            \
    }                                                                       \
  } __name;                                                                 \
  const char *DynLoadShim__##__name::kName = #__name;

PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandCreateGenerator);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandDestroyGenerator);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandSetStream);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandGenerateUniform);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandGenerateUniformDouble);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandSetPseudoRandomGeneratorSeed);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandSetGeneratorOffset);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandGenerateNormal);
PERFTOOLS_GPUTOOLS_CURAND_WRAP(curandGenerateNormalDouble);

}  // namespace dynload

template <typename T>
string TypeString();

template <>
string TypeString<float>() {
  return "float";
}

template <>
string TypeString<double>() {
  return "double";
}

template <>
string TypeString<std::complex<float>>() {
  return "std::complex<float>";
}

template <>
string TypeString<std::complex<double>>() {
  return "std::complex<double>";
}

CUDARng::CUDARng(CUDAExecutor *parent) : parent_(parent), rng_(nullptr) {}

CUDARng::~CUDARng() {
  if (rng_ != nullptr) {
    dynload::curandDestroyGenerator(parent_, rng_);
  }
}

bool CUDARng::Init() {
  mutex_lock lock{mu_};
  CHECK(rng_ == nullptr);

  curandStatus_t ret =
      dynload::curandCreateGenerator(parent_, &rng_, CURAND_RNG_PSEUDO_DEFAULT);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create random number generator: " << ret;
    return false;
  }

  CHECK(rng_ != nullptr);
  return true;
}

<<<<<<< HEAD
bool GpuRng::SetStream(Stream* stream) {
  cuda::ScopedActivateExecutorContext sac(parent_);
  curandStatus_t ret = curandSetStream(rng_, AsGpuStreamValue(stream));
=======
bool CUDARng::SetStream(Stream *stream) {
  curandStatus_t ret =
      dynload::curandSetStream(parent_, rng_, AsCUDAStreamValue(stream));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for random generation: " << ret;
    return false;
  }

  return true;
}

// Returns true if std::complex stores its contents as two consecutive
// elements. Tests int, float and double, as the last two are independent
// specializations.
constexpr bool ComplexIsConsecutiveFloats() {
  return sizeof(std::complex<int>) == 8 && sizeof(std::complex<float>) == 8 &&
      sizeof(std::complex<double>) == 16;
}

template <typename T>
<<<<<<< HEAD
bool GpuRng::DoPopulateRandUniformInternal(Stream* stream, DeviceMemory<T>* v) {
  absl::MutexLock lock(&mu_);
=======
bool CUDARng::DoPopulateRandUniformInternal(Stream *stream,
                                            DeviceMemory<T> *v) {
  mutex_lock lock{mu_};
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  static_assert(ComplexIsConsecutiveFloats(),
                "std::complex values are not stored as consecutive values");

  if (!SetStream(stream)) {
    return false;
  }

  // std::complex<T> is currently implemented as two consecutive T variables.
  uint64 element_count = v->ElementCount();
  if (std::is_same<T, std::complex<float>>::value ||
      std::is_same<T, std::complex<double>>::value) {
    element_count *= 2;
  }

<<<<<<< HEAD
  cuda::ScopedActivateExecutorContext sac(parent_);
  curandStatus_t ret;
  if (std::is_same<T, float>::value ||
      std::is_same<T, std::complex<float>>::value) {
    ret = curandGenerateUniform(
        rng_, reinterpret_cast<float*>(GpuMemoryMutable(v)), element_count);
  } else {
    ret = curandGenerateUniformDouble(
        rng_, reinterpret_cast<double*>(GpuMemoryMutable(v)), element_count);
=======
  curandStatus_t ret;
  if (std::is_same<T, float>::value ||
      std::is_same<T, std::complex<float>>::value) {
    ret = dynload::curandGenerateUniform(
        parent_, rng_, reinterpret_cast<float *>(CUDAMemoryMutable(v)),
        element_count);
  } else {
    ret = dynload::curandGenerateUniformDouble(
        parent_, rng_, reinterpret_cast<double *>(CUDAMemoryMutable(v)),
        element_count);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do uniform generation of " << v->ElementCount()
               << " " << TypeString<T>() << "s at " << v->opaque() << ": "
               << ret;
    return false;
  }

  return true;
}

<<<<<<< HEAD
bool GpuRng::DoPopulateRandUniform(Stream* stream, DeviceMemory<float>* v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream, DeviceMemory<double>* v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream,
                                   DeviceMemory<std::complex<float>>* v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream,
                                   DeviceMemory<std::complex<double>>* v) {
=======
bool CUDARng::DoPopulateRandUniform(Stream *stream, DeviceMemory<float> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool CUDARng::DoPopulateRandUniform(Stream *stream, DeviceMemory<double> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool CUDARng::DoPopulateRandUniform(Stream *stream,
                                    DeviceMemory<std::complex<float>> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool CUDARng::DoPopulateRandUniform(Stream *stream,
                                    DeviceMemory<std::complex<double>> *v) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return DoPopulateRandUniformInternal(stream, v);
}

template <typename ElemT, typename FuncT>
<<<<<<< HEAD
bool GpuRng::DoPopulateRandGaussianInternal(Stream* stream, ElemT mean,
                                            ElemT stddev,
                                            DeviceMemory<ElemT>* v,
                                            FuncT func) {
  absl::MutexLock lock(&mu_);
=======
bool CUDARng::DoPopulateRandGaussianInternal(Stream *stream, ElemT mean,
                                             ElemT stddev,
                                             DeviceMemory<ElemT> *v,
                                             FuncT func) {
  mutex_lock lock{mu_};
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  if (!SetStream(stream)) {
    return false;
  }

<<<<<<< HEAD
  cuda::ScopedActivateExecutorContext sac(parent_);
  uint64 element_count = v->ElementCount();
  curandStatus_t ret =
      func(rng_, GpuMemoryMutable(v), element_count, mean, stddev);
=======
  uint64 element_count = v->ElementCount();
  curandStatus_t ret =
      func(parent_, rng_, CUDAMemoryMutable(v), element_count, mean, stddev);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do gaussian generation of " << v->ElementCount()
               << " floats at " << v->opaque() << ": " << ret;
    return false;
  }

  return true;
}

<<<<<<< HEAD
bool GpuRng::DoPopulateRandGaussian(Stream* stream, float mean, float stddev,
                                    DeviceMemory<float>* v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        curandGenerateNormal);
}

bool GpuRng::DoPopulateRandGaussian(Stream* stream, double mean, double stddev,
                                    DeviceMemory<double>* v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        curandGenerateNormalDouble);
}

bool GpuRng::SetSeed(Stream* stream, const uint8* seed, uint64 seed_bytes) {
  absl::MutexLock lock(&mu_);
=======
bool CUDARng::DoPopulateRandGaussian(Stream *stream, float mean, float stddev,
                                     DeviceMemory<float> *v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        dynload::curandGenerateNormal);
}

bool CUDARng::DoPopulateRandGaussian(Stream *stream, double mean, double stddev,
                                     DeviceMemory<double> *v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        dynload::curandGenerateNormalDouble);
}

bool CUDARng::SetSeed(Stream *stream, const uint8 *seed, uint64 seed_bytes) {
  mutex_lock lock{mu_};
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  CHECK(rng_ != nullptr);

  if (!CheckSeed(seed, seed_bytes)) {
    return false;
  }

  if (!SetStream(stream)) {
    return false;
  }

<<<<<<< HEAD
  cuda::ScopedActivateExecutorContext sac(parent_);
  // Requires 8 bytes of seed data; checked in RngSupport::CheckSeed (above)
  // (which itself requires 16 for API consistency with host RNG fallbacks).
  curandStatus_t ret = curandSetPseudoRandomGeneratorSeed(
      rng_, *(reinterpret_cast<const uint64*>(seed)));
=======
  // Requires 8 bytes of seed data; checked in RngSupport::CheckSeed (above)
  // (which itself requires 16 for API consistency with host RNG fallbacks).
  curandStatus_t ret = dynload::curandSetPseudoRandomGeneratorSeed(
      parent_, rng_, *(reinterpret_cast<const uint64 *>(seed)));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set rng seed: " << ret;
    return false;
  }

<<<<<<< HEAD
  ret = curandSetGeneratorOffset(rng_, 0);
=======
  ret = dynload::curandSetGeneratorOffset(parent_, rng_, 0);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to reset rng position: " << ret;
    return false;
  }
  return true;
}

<<<<<<< HEAD
}  // namespace gpu

void initialize_curand() {
  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::RngFactory>(
          cuda::kCudaPlatformId, gpu::kGpuRandPlugin, "cuRAND",
          [](internal::StreamExecutorInterface* parent) -> rng::RngSupport* {
            gpu::GpuExecutor* cuda_executor =
                dynamic_cast<gpu::GpuExecutor*>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR)
                  << "Attempting to initialize an instance of the cuRAND "
                  << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            gpu::GpuRng* rng = new gpu::GpuRng(cuda_executor);
            if (!rng->Init()) {
              // Note: Init() will log a more specific error.
              delete rng;
              return nullptr;
            }
            return rng;
          });
=======
}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

namespace gpu = ::perftools::gputools;

REGISTER_MODULE_INITIALIZER(register_curand, {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::RngFactory>(
              gpu::cuda::kCudaPlatformId, gpu::cuda::kCuRandPlugin, "cuRAND",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::rng::RngSupport * {
                gpu::cuda::CUDAExecutor *cuda_executor =
                    dynamic_cast<gpu::cuda::CUDAExecutor *>(parent);
                if (cuda_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the cuRAND "
                      << "support library with a non-CUDA StreamExecutor";
                  return nullptr;
                }

                gpu::cuda::CUDARng *rng = new gpu::cuda::CUDARng(cuda_executor);
                if (!rng->Init()) {
                  // Note: Init() will log a more specific error.
                  delete rng;
                  return nullptr;
                }
                return rng;
              });
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuRAND factory: "
               << status.error_message();
  }

<<<<<<< HEAD
  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kRng, gpu::kGpuRandPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_curand,
                            { stream_executor::initialize_curand(); });
=======
  // Prime the cuRAND DSO. The loader will log more information.
  auto statusor = gpu::internal::CachedDsoLoader::GetCurandDsoHandle();
  if (!statusor.ok()) {
    LOG(INFO) << "Unable to load cuRAND DSO.";
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kRng,
                                                     gpu::cuda::kCuRandPlugin);
});
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
