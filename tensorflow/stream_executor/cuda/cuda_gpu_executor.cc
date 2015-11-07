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

#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#define PATH_MAX MAX_PATH
#else
#include <unistd.h>
#endif
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_event.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_timer.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
=======
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"

#include <unistd.h>

#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_event.h"
#include "tensorflow/stream_executor/cuda/cuda_platform.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_timer.h"
#include "tensorflow/stream_executor/dso_loader.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/lib/casts.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/mathutil.h"
<<<<<<< HEAD
#include "tensorflow/stream_executor/lib/numbers.h"
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/statusor.h"
=======
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"
<<<<<<< HEAD

// LOG(ERROR) uses a const named ERROR, so a macro with the same name is
// always unwanted. This happens on Windows that defines such a macro.
#undef ERROR
=======
#include "tensorflow/stream_executor/lib/numbers.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#ifdef PLATFORMS_GPUS_CUDA_DYNAMIC_LIBCUDA_DYNAMIC_LIBCUDA_H_
#error \
    "No driver calls in this file, wrap driver functionality in cuda_driver.cc."
#endif

#ifdef __CUDA_RUNTIME_H__
#error \
    "CUDA runtime being included into CUDA GPU executor; should be driver only."
#endif

extern bool FLAGS_check_gpu_leaks;
<<<<<<< HEAD
bool FLAGS_prefer_cubin_to_ptx = true;

namespace stream_executor {
namespace gpu {
=======
tensorflow::int32 FLAGS_register_occupancy_warning_threshold;
bool FLAGS_prefer_cubin_to_ptx = true;

namespace perftools {
namespace gputools {
namespace rng {
class RngSupport;
}  // namespace rng
}  // namespace gputools
}  // namespace perftools

namespace perftools {
namespace gputools {
namespace cuda {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// Hook that can be used to CUBIN-ate PTX before it is loaded into the driver.
// It has been observed that loading both PTX and cubins into the driver library
// can cause it to crash, but loading only CUBINs avoids those crashes;
// therefore, it's useful to have this hook to hack in uniform CUBIN-ation of
// PTX code.
//
// As this is an implementation-detail workaround, the usage is to declare this
// variable with extern linkage and populate it from another translation unit.
std::function<string(const string &)> g_cubinate;

<<<<<<< HEAD
static GpuEvent* AsGpuEvent(Event* event) {
  DCHECK(event != nullptr);
  return static_cast<GpuEvent*>(event->implementation());
=======
static CUDAEvent *AsCUDAEvent(Event *event) {
  DCHECK(event != nullptr);
  return static_cast<CUDAEvent *>(event->implementation());
}

// Given a platform-independent stream datatype, returns the internal CUDA
// platform implementation pointer.
static CUDAStream *AsCUDAStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return static_cast<CUDAStream *>(stream->implementation());
}

// Given a platform-independent stream datatype, returns the platform
// implementation's internal value, suitable for passing directly to libcuda
// APIs.
CUstream AsCUDAStreamValue(Stream *stream) {
  DCHECK(stream != nullptr);
  return AsCUDAStream(stream)->cuda_stream();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

// Given a platform-independent timer datatype, returns the internal CUDA
// platform implementation pointer.
<<<<<<< HEAD
static GpuTimer* AsGpuTimer(Timer* timer) {
  DCHECK(timer != nullptr);
  return static_cast<GpuTimer*>(timer->implementation());
=======
static CUDATimer *AsCUDATimer(Timer *timer) {
  DCHECK(timer != nullptr);
  return static_cast<CUDATimer *>(timer->implementation());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

// Given const GPU memory, returns a libcuda device pointer datatype, suitable
// for passing directly to libcuda APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// libcuda APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to libcuda functions which will honor constness.
static CUdeviceptr AsCudaDevicePtr(const DeviceMemoryBase &gpu_mem) {
  return reinterpret_cast<CUdeviceptr>(gpu_mem.opaque());
}

// See description on const version above.
static CUdeviceptr AsCudaDevicePtr(DeviceMemoryBase *gpu_mem) {
  return AsCudaDevicePtr(*gpu_mem);
}

<<<<<<< HEAD
GpuContext* ExtractGpuContext(GpuExecutor* cuda_exec) {
  CHECK(cuda_exec != nullptr);
  return cuda_exec->gpu_context();
}

GpuExecutor* ExtractGpuExecutor(StreamExecutor* stream_exec) {
  return static_cast<GpuExecutor*>(stream_exec->implementation());
}

GpuExecutor::~GpuExecutor() {
  CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
  if (context_ != nullptr) {
    GpuDriver::DestroyContext(context_);
  }
}

port::Status GpuExecutor::Init(int device_ordinal,
                               DeviceOptions device_options) {
  device_ordinal_ = device_ordinal;

  auto status = GpuDriver::Init();
=======
static CUcontext GetCudaContext(Stream *stream) {
  return static_cast<CUDAExecutor *>(stream->parent()->implementation())
      ->cuda_context();
}

CUcontext ExtractCudaContext(CUDAExecutor *cuda_exec) {
  CHECK(cuda_exec != nullptr);
  return cuda_exec->cuda_context();
}

CUDAExecutor *ExtractCudaExecutor(StreamExecutor *stream_exec) {
  return static_cast<CUDAExecutor *>(stream_exec->implementation());
}

CUDAExecutor::~CUDAExecutor() {
  for (auto &it : disk_modules_) {
    CUDADriver::UnloadModule(context_, it.second);
  }
  for (auto &it : in_memory_modules_) {
    CUDADriver::UnloadModule(context_, it.second);
  }
  if (context_ != nullptr) {
    CUDADriver::DestroyContext(context_);
  }
}

port::Status CUDAExecutor::Init(int device_ordinal,
                                DeviceOptions device_options) {
  device_ordinal_ = device_ordinal;

  auto status = CUDADriver::Init();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!status.ok()) {
    return status;
  }

<<<<<<< HEAD
  status = GpuDriver::GetDevice(device_ordinal_, &device_);
=======
  status = CUDADriver::GetDevice(device_ordinal_, &device_);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!status.ok()) {
    return status;
  }

<<<<<<< HEAD
  status = GpuDriver::CreateContext(device_ordinal_, device_, device_options,
                                    &context_);
=======
  status = CUDADriver::CreateContext(device_, device_options, &context_);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!status.ok()) {
    return status;
  }

<<<<<<< HEAD
  return GpuDriver::GetComputeCapability(&cc_major_, &cc_minor_, device_);
}

bool GpuExecutor::FindOnDiskForComputeCapability(
    absl::string_view filename, absl::string_view canonical_suffix,
    string* found_filename) const {
=======
  return CUDADriver::GetComputeCapability(&cc_major_, &cc_minor_, device_);
}

bool CUDAExecutor::FindOnDiskForComputeCapability(
    port::StringPiece filename, port::StringPiece canonical_suffix,
    string *found_filename) const {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (cc_major_ == 0 && cc_minor_ == 0) {
    return false;
  }

<<<<<<< HEAD
  string cc_specific =
      absl::StrCat(filename, ".cc", cc_major_, cc_minor_, canonical_suffix);
  if (port::FileExists(cc_specific).ok()) {
=======
  // TODO(22689637): Eliminate unnecessary ToString()s when all dependencies
  // have been migrated.
  string cc_specific = port::StrCat(filename.ToString(), ".cc", cc_major_,
                                    cc_minor_, canonical_suffix.ToString());
  if (port::FileExists(cc_specific)) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    VLOG(2) << "found compute-capability-specific file, using that: "
            << cc_specific;
    *found_filename = cc_specific;
    return true;
  }

  VLOG(2) << "could not find compute-capability specific file at: "
          << cc_specific;
<<<<<<< HEAD
  if (port::FileExists(string(filename)).ok()) {
    *found_filename = string(filename);
=======
  if (port::FileExists(filename.ToString())) {
    *found_filename = filename.ToString();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    return true;
  }

  return false;
}

<<<<<<< HEAD
bool GpuExecutor::FindOnDiskForISAVersion(absl::string_view filename,
                                          absl::string_view canonical_suffix,
                                          string* found_filename) const {
  LOG(ERROR)
      << "Feature not supported on CUDA platform (FindOnDiskForISAVersion)";
  return false;
}
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// Returns the path to the running executable.
// N.B. Derived from //knowledge/smalltalk/background_kb.cc
// Arg: strip_exe: if true, remove the name of the executable itself from the
//                 returned string. Example: calling this from /usr/bin/foo
//                 would return /usr/bin.
static string GetBinaryDir(bool strip_exe) {
  char exe_path[PATH_MAX] = {0};
<<<<<<< HEAD
#if defined(__APPLE__)
  uint32_t buffer_size = 0U;
  _NSGetExecutablePath(nullptr, &buffer_size);
  char unresolved_path[buffer_size];
  _NSGetExecutablePath(unresolved_path, &buffer_size);
  CHECK_ERR(realpath(unresolved_path, exe_path) ? 1 : -1);
#else
#if defined(PLATFORM_WINDOWS)
  HMODULE hModule = GetModuleHandle(NULL);
  GetModuleFileName(hModule, exe_path, MAX_PATH);
#else
  PCHECK(readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1) != -1);
#endif
#endif
=======
  CHECK_ERR(readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // Make sure it's null-terminated:
  exe_path[sizeof(exe_path) - 1] = 0;

  if (strip_exe) {
    // The exe is the last component of the path, so remove one component.
    string ret = exe_path;
<<<<<<< HEAD
    std::vector<string> components = absl::StrSplit(exe_path, '/');
    components.pop_back();
    return absl::StrJoin(components, "/");
=======
    std::vector<string> components = port::Split(exe_path, '/');
    components.pop_back();
    return port::Join(components, "/");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
  return exe_path;
}

<<<<<<< HEAD
port::Status GpuExecutor::LoadModuleFromCuBin(const char* cubin,
                                              CUmodule* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[cubin];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(GpuDriver::LoadCubin(context_, cubin, module));
    module_refcount = 1;
    VLOG(3) << "Loaded CUBIN " << static_cast<const void *>(cubin)
            << " as module " << *module;
  } else {
    ++module_refcount;
    VLOG(3) << "CUBIN " << static_cast<const void *>(cubin)
            << " is already loaded as module " << *module;
  }
  gpu_binary_to_module_[cubin] = {*module, module_refcount};
  return port::Status::OK();
}

port::Status GpuExecutor::LoadModuleFromPtx(const char* ptx, CUmodule* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[ptx];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(GpuDriver::LoadPtx(context_, ptx, module));
    VLOG(3) << "Loaded PTX " << static_cast<const void *>(ptx) << " as module "
            << *module;
    module_refcount = 1;
  } else {
    ++module_refcount;
    VLOG(3) << "PTX " << static_cast<const void *>(ptx)
            << " is already loaded as module " << module;
  }
  gpu_binary_to_module_[ptx] = {*module, module_refcount};
  return port::Status::OK();
}

port::Status GpuExecutor::LoadModuleFromHsaco(const char* hsaco,
                                              CUmodule* module) {
  return port::InternalError(
      "Feature not supported on CUDA platform (LoadModuleFromHsaco)");
}

port::Status GpuExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                    KernelBase* kernel) {
  GpuKernel* cuda_kernel = AsGpuKernel(kernel);
  CUmodule module;
  const string *kernelname;

  VLOG(3) << "GetKernel on kernel " << kernel << " : " << kernel->name();

  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    kernelname = &spec.cuda_cubin_in_memory().kernelname();
    const char *cubin = spec.cuda_cubin_in_memory().bytes();
    TF_RETURN_IF_ERROR(LoadModuleFromCuBin(cubin, &module));
    kernel_to_gpu_binary_[kernel] = cubin;
=======
// Returns the location of the runfiles directory.
// This is the directory which "bazel run" sets as the current working directory
// before the program starts.
// N.B. This doesn't have to be running under "bazel run" in order to get the
// appropriate runfiles directory.
static string GetRunfilesDir() {
  return port::StrCat(GetBinaryDir(false), ".runfiles");
}

bool CUDAExecutor::GetKernel(const MultiKernelLoaderSpec &spec,
                             KernelBase *kernel) {
  CUDAKernel *cuda_kernel = AsCUDAKernel(kernel);
  CUmodule module = nullptr;
  const string *kernelname;

  const OnDiskKernelLoaderSpec *on_disk_spec = nullptr;
  bool has_ptx = spec.has_cuda_ptx_on_disk();
  bool has_cubin = spec.has_cuda_cubin_on_disk();
  if (has_cubin && (!has_ptx || FLAGS_prefer_cubin_to_ptx)) {
    on_disk_spec = &spec.cuda_cubin_on_disk();
  } else if (has_ptx) {
    on_disk_spec = &spec.cuda_ptx_on_disk();
  }

  if (on_disk_spec != nullptr) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  } else if (spec.has_cuda_ptx_in_memory()) {
    kernelname = &spec.cuda_ptx_in_memory().kernelname();

    if (cc_major_ == 0 && cc_minor_ == 0) {
<<<<<<< HEAD
      return port::InternalError("Compute capability not set");
    }

    const char *ptx = spec.cuda_ptx_in_memory().text(cc_major_, cc_minor_);
    if (ptx == nullptr) {
      ptx = spec.cuda_ptx_in_memory().default_text();
    }
    if (ptx == nullptr) {
      LOG(FATAL) << "Loader spec has no ptx for kernel " << *kernelname;
    }

    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(LoadModuleFromPtx(ptx, &module));
    kernel_to_gpu_binary_[kernel] = ptx;
  } else {
    return port::InternalError("No method of loading CUDA kernel provided");
  }
  VLOG(2) << "getting function " << *kernelname << " from module " << module;
  if (!GpuDriver::GetModuleFunction(context_, module, kernelname->c_str(),
                                    cuda_kernel->gpu_function_ptr())) {
    return port::InternalError("Could not find the corresponding function");
=======
      return false;
    }

    // Note that the orignal ptx may be compressed, and the ptx we get below is
    // the decompressed result. To cache the module we should use the original
    // ptx (compressed one) as the key. This is because for the same compressed
    // ptx, we may get different decompressed ptx wrt the pointer value.
    const char *ptx = spec.cuda_ptx_in_memory().text(cc_major_, cc_minor_);
    const char *orig_ptx =
        spec.cuda_ptx_in_memory().original_text(cc_major_, cc_minor_);
    if (ptx == nullptr || orig_ptx == nullptr) {
      ptx = spec.cuda_ptx_in_memory().default_text();
      orig_ptx = spec.cuda_ptx_in_memory().original_default_text();
    }
    if (ptx == nullptr || orig_ptx == nullptr) {
      LOG(FATAL) << "could not load ptx for kernel " << kernelname;
      return false;
    }

    mutex_lock lock{in_memory_modules_mu_};
    module = in_memory_modules_[orig_ptx];

    if (module == nullptr) {
      if (g_cubinate == nullptr) {
        if (!CUDADriver::LoadPtx(context_, ptx, &module)) {
          return false;
        }
      } else {
        string cubin = g_cubinate(ptx);
        auto load_status =
            CUDADriver::LoadCubin(context_, cubin.c_str(), &module);
        if (!load_status.ok()) {
          LOG(ERROR) << "failed to load cubin via hook: " << load_status;
          return false;
        }
      }
      in_memory_modules_[orig_ptx] = module;
    }
  } else if (spec.has_cuda_cubin_in_memory()) {
    kernelname = &spec.cuda_cubin_in_memory().kernelname();
    const char *cubin = spec.cuda_cubin_in_memory().bytes();
    mutex_lock lock{in_memory_modules_mu_};
    module = in_memory_modules_[cubin];

    if (module == nullptr) {
      auto load_status = CUDADriver::LoadCubin(context_, cubin, &module);
      if (!load_status.ok()) {
        LOG(ERROR) << "failed to load CUBIN: " << load_status;
        return false;
      }

      in_memory_modules_[cubin] = module;
    }
  } else {
    LOG(WARNING) << "no method of loading CUDA kernel provided";
    return false;
  }

  VLOG(2) << "getting function " << kernelname << " from module " << module;
  if (!CUDADriver::GetModuleFunction(context_, module, kernelname->c_str(),
                                     cuda_kernel->cuda_function_ptr())) {
    return false;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the CUDA API.
  cuda_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
<<<<<<< HEAD
  TF_RETURN_IF_ERROR(GetKernelMetadata(cuda_kernel, &kernel_metadata));
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(*kernelname);
  return port::Status::OK();
}

bool GpuExecutor::UnloadGpuBinary(const void* gpu_binary) {
  auto module_it = gpu_binary_to_module_.find(gpu_binary);
  if (gpu_binary_to_module_.end() == module_it) {
    VLOG(3) << "No loaded CUDA module for " << gpu_binary;
    return false;
  }
  auto &module = module_it->second.first;
  auto &refcount = module_it->second.second;
  VLOG(3) << "Found CUDA module " << module << " with refcount " << refcount;
  if (--refcount == 0) {
    VLOG(3) << "Unloading CUDA module " << module;
    GpuDriver::UnloadModule(context_, module);
    gpu_binary_to_module_.erase(module_it);
  }
  return true;
}

void GpuExecutor::UnloadKernel(const KernelBase* kernel) {
  VLOG(3) << "Unloading kernel " << kernel << " : " << kernel->name();

  absl::MutexLock lock{&in_memory_modules_mu_};
  auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
  if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
    VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
            << " has never been loaded.";
    return;  // We've never seen this kernel.
  }
  VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
          << " has loaded GPU code " << gpu_binary_it->second;
  UnloadGpuBinary(gpu_binary_it->second);
  kernel_to_gpu_binary_.erase(gpu_binary_it);
}

port::Status GpuExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                     ModuleHandle* module_handle) {
  // In GpuExecutor we store the pointer to the GPU binary (PTX or CUBIN) as
  // ModuleHandle::id().
  CUmodule cu_module;
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(LoadModuleFromCuBin(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()),
        &cu_module));
    *module_handle = ModuleHandle(const_cast<void *>(
        static_cast<const void *>(spec.cuda_cubin_in_memory().data())));
    return port::Status::OK();
  } else if (spec.has_cuda_ptx_in_memory()) {
    if (cc_major_ == 0 && cc_minor_ == 0) {
      return port::InternalError("Compute capability not set");
    }

    if (!spec.cuda_ptx_in_memory()) {
      return port::InternalError("PTX not found in spec");
    }

    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(
        LoadModuleFromPtx(spec.cuda_ptx_in_memory(), &cu_module));
    *module_handle = ModuleHandle(const_cast<void *>(
        static_cast<const void *>(spec.cuda_ptx_in_memory())));
    return port::Status::OK();
  }
  return port::InternalError("No method of loading CUDA module provided");
}

bool GpuExecutor::UnloadModule(ModuleHandle module_handle) {
  const char *gpu_binary = reinterpret_cast<const char *>(module_handle.id());
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(gpu_binary);
}

port::Status GpuExecutor::GetKernelMetadata(GpuKernel* cuda_kernel,
                                            KernelMetadata* kernel_metadata) {
  int value;
  TF_RETURN_IF_ERROR(GpuDriver::FuncGetAttribute(
      CU_FUNC_ATTRIBUTE_NUM_REGS, *cuda_kernel->gpu_function_ptr(), &value));
  kernel_metadata->set_registers_per_thread(value);

  TF_RETURN_IF_ERROR(
      GpuDriver::FuncGetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                  *cuda_kernel->gpu_function_ptr(), &value));
  kernel_metadata->set_shared_memory_bytes(value);
  return port::Status::OK();
}

port::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const KernelBase& kernel,
                                 const KernelArgsArrayBase& args) {
  CHECK_EQ(kernel.Arity(), args.number_of_arguments());
  CUstream custream = AsGpuStreamValue(stream);
  const GpuKernel* cuda_kernel = AsGpuKernel(&kernel);
  CUfunction cufunc = cuda_kernel->AsGpuFunctionHandle();

  // Only perform/print the occupancy check once.  Even just checking to see
  // whether we've done an occupancy check on this kernel before isn't free
  // (because we have to synchronize), so we only do this at -v 2+.
  if (VLOG_IS_ON(2)) {
    absl::MutexLock lock(&launched_kernels_mu_);
    if (!launched_kernels_.count(cufunc)) {
      VlogOccupancyInfo(kernel, thread_dims, block_dims);
      // TODO(rspringer): Remove elements from launched_kernels_...if we ever
      // expose a kernel/module deallocation method.
      launched_kernels_.insert(cufunc);
    }
  }

  if (cuda_kernel->GetPreferredCacheConfig() !=
      KernelCacheConfig::kNoPreference) {
    TF_RETURN_IF_ERROR(GpuDriver::FuncSetCacheConfig(
        cufunc, cuda_kernel->GetGpuCacheConfig()));
  }

  void **kernel_params = const_cast<void **>(args.argument_addresses().data());

  return GpuDriver::LaunchKernel(
      context_, cufunc, block_dims.x, block_dims.y, block_dims.z, thread_dims.x,
      thread_dims.y, thread_dims.z, args.number_of_shared_bytes(), custream,
      kernel_params, nullptr /* = extra */);
=======
  if (!GetKernelMetadata(cuda_kernel, &kernel_metadata)) {
    LOG(WARNING) << "Unable to get metadata for kernel " << kernelname;
  }
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(*kernelname);
  return true;
}

bool CUDAExecutor::GetKernelMetadata(CUDAKernel *cuda_kernel,
                                     KernelMetadata *kernel_metadata) {
  int value;
  if (!CUDADriver::FuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS,
                                    *cuda_kernel->cuda_function_ptr(),
                                    &value)) {
    return false;
  }
  kernel_metadata->set_registers_per_thread(value);

  if (!CUDADriver::FuncGetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                    *cuda_kernel->cuda_function_ptr(),
                                    &value)) {
    return false;
  }
  kernel_metadata->set_shared_memory_bytes(value);

  return true;
}

bool CUDAExecutor::Launch(Stream *stream, const ThreadDim &thread_dims,
                          const BlockDim &block_dims, const KernelBase &kernel,
                          const std::vector<KernelArg> &args) {
  CHECK_EQ(kernel.Arity(), args.size());
  CUstream custream = AsCUDAStreamValue(stream);
  const CUDAKernel *cuda_kernel = AsCUDAKernel(&kernel);
  CUfunction cufunc = cuda_kernel->AsCUDAFunctionValue();

  std::vector<void *> addrs;
  addrs.reserve(args.size());
  int shmem_bytes = 0;
  for (size_t i = 0; i < args.size(); i++) {
    switch (args[i].type) {
      case KernelArg::kNormal:
        addrs.push_back(const_cast<void *>(
            static_cast<const void *>(args[i].data.begin())));
        break;
      case KernelArg::kSharedMemory:
        shmem_bytes += args[i].bytes;
        break;
      default:
        LOG(ERROR) << "Invalid kernel arg type passed (" << args[i].type
                   << ") for arg " << i;
        return false;
    }
  }

  // Only perform/print the occupancy check 1x.
  launched_kernels_mu_.lock();
  if (launched_kernels_.find(cufunc) == launched_kernels_.end()) {
    OccupancyCheck(kernel, thread_dims, block_dims);
    // TODO(rspringer): Remove elements from launched_kernels_...if we ever
    // expose a kernel/module deallocation method.
    launched_kernels_.insert(cufunc);
  }
  launched_kernels_mu_.unlock();

  if (cuda_kernel->GetPreferredCacheConfig() !=
      KernelCacheConfig::kNoPreference) {
    CUDADriver::FuncSetCacheConfig(cufunc, cuda_kernel->GetCUDACacheConfig());
  }

  if (!CUDADriver::LaunchKernel(
          GetCudaContext(stream), cufunc, block_dims.x, block_dims.y,
          block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
          shmem_bytes, custream, addrs.data(), nullptr /* = extra */)) {
    LOG(ERROR) << "failed to launch CUDA kernel with args: " << args.size()
               << "; thread dim: " << thread_dims.ToString()
               << "; block dim: " << block_dims.ToString();
    return false;
  }

  return true;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

// This is a non-essential operation; if there's a failure, proceed without
// logging an error. It's nearly certain that in case of failures, we'd never
// get here in the first place; these are very low-impact routines.
<<<<<<< HEAD
void GpuExecutor::VlogOccupancyInfo(const KernelBase& kernel,
                                    const ThreadDim& thread_dims,
                                    const BlockDim& block_dims) {
=======
void CUDAExecutor::OccupancyCheck(const KernelBase &kernel,
                                  const ThreadDim &thread_dims,
                                  const BlockDim &block_dims) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  VLOG(2) << "Computing kernel occupancy for kernel "
          << kernel.demangled_name();
  VLOG(2) << "Thread dimensions (" << thread_dims.x << ", " << thread_dims.y
          << ", " << thread_dims.z << ")";

  int regs_per_thread;
  if (!kernel.metadata().registers_per_thread(&regs_per_thread)) {
    return;
  }

  int smem_per_block;
  if (!kernel.metadata().shared_memory_bytes(&smem_per_block)) {
    return;
  }

  const DeviceDescription &device_description =
      kernel.parent()->GetDeviceDescription();

<<<<<<< HEAD
  const GpuKernel* cuda_kernel = AsGpuKernel(&kernel);
  CUfunction cufunc = cuda_kernel->AsGpuFunctionHandle();

  int blocks_per_sm = CalculateOccupancy(device_description, regs_per_thread,
                                         smem_per_block, thread_dims, cufunc);
  VLOG(2) << "Resident blocks per SM is " << blocks_per_sm;

  int suggested_threads =
      CompareOccupancy(&blocks_per_sm, device_description, regs_per_thread,
                       smem_per_block, thread_dims, cufunc);
  if (suggested_threads != 0) {
    VLOG(2) << "The cuda occupancy calculator recommends using "
            << suggested_threads
            << " threads per block to achieve an occupancy of " << blocks_per_sm
            << " blocks per SM.";
  }
}

// Compute and return maximum blocks per core (occupancy) based on the
// device description, some kernel characteristics and the number of threads per
// block.  If unable to compute occupancy, zero is returned.
int GpuExecutor::CalculateOccupancy(const DeviceDescription& device_description,
                                    uint64 registers_per_thread,
                                    uint64 shared_memory_per_block,
                                    const ThreadDim& thread_dims,
                                    CUfunction func) {
  int suggested_blocks = 0;
  int suggested_threads = 0;
  CUresult err = cuOccupancyMaxPotentialBlockSize(
      &suggested_blocks, &suggested_threads, func, nullptr,
      shared_memory_per_block, 0);
  CHECK_EQ(err, CUDA_SUCCESS);
  return suggested_blocks;
}

// Compute and return the suggested thread count to achieve ideal occupancy.
// If the provided thread dimensions match this number, zero is returned.
int GpuExecutor::CompareOccupancy(int* initial_blocks,
                                  const DeviceDescription& device_description,
                                  uint64 registers_per_thread,
                                  uint64 shared_memory_per_block,
                                  const ThreadDim& thread_dims,
                                  CUfunction func) {
  int suggested_blocks = 0;
  int suggested_threads = 0;
  CUresult err = cuOccupancyMaxPotentialBlockSize(
      &suggested_blocks, &suggested_threads, func, nullptr,
      shared_memory_per_block, 0);
  CHECK_EQ(err, CUDA_SUCCESS);
  if (suggested_blocks > *initial_blocks) {
    *initial_blocks = suggested_blocks;
    return suggested_threads;
  } else {
    return 0;
  }
}

DeviceMemoryBase GpuExecutor::Allocate(uint64 size, int64 memory_space) {
  CHECK_EQ(memory_space, 0);
  return DeviceMemoryBase(GpuDriver::DeviceAllocate(context_, size), size);
}

void* GpuExecutor::GetSubBuffer(DeviceMemoryBase* mem, uint64 offset_bytes,
                                uint64 size_bytes) {
=======
  uint64 blocks_per_sm = CalculateOccupancy(
      device_description, regs_per_thread, smem_per_block, thread_dims);
  VLOG(2) << "Resident blocks per SM is " << blocks_per_sm;

  // To increase occupancy, there must be a sufficient number of blocks
  // available to spread across the sm's at this new improved occupancy level.
  int multiprocessor_count = device_description.core_count();
  int block_count = block_dims.x * block_dims.y * block_dims.z;
  int available_blocks_per_sm =
      port::MathUtil::CeilOfRatio(block_count, multiprocessor_count);
  if (available_blocks_per_sm <= static_cast<int64>(blocks_per_sm)) {
    VLOG(2) << "Occupancy is limited by number of blocks available per sm.";
    return;
  }

  uint64 improved_regs_per_thread = CalculateRegisterLimitForTargetOccupancy(
      device_description, smem_per_block, thread_dims, blocks_per_sm + 1);
  if (improved_regs_per_thread != 0) {
    VLOG(2) << "Reducing register usage from " << regs_per_thread
            << " to " << improved_regs_per_thread
            << " could increase resident blocks per SM by one.";

    uint64 reg_reduction = regs_per_thread - improved_regs_per_thread;
    if (reg_reduction <=
        static_cast<uint64>(FLAGS_register_occupancy_warning_threshold)) {
      LOG(INFO) << "Notice: occupancy would increase if register usage was"
                << " reduced from " << regs_per_thread
                << " to " << improved_regs_per_thread
                << " registers per thread for kernel: "
                << kernel.demangled_name();
    }
  } else {
    VLOG(2) << "Resident blocks per SM cannot be increased by reducing "
        "register usage.";
  }
}

void *CUDAExecutor::Allocate(uint64 size) {
  return CUDADriver::DeviceAllocate(context_, size);
}

void *CUDAExecutor::AllocateSubBuffer(DeviceMemoryBase *mem,
                                      uint64 offset_bytes, uint64 size_bytes) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // offset and size are in bytes, so char* works as the pointer type.
  return reinterpret_cast<char *>(mem->opaque()) + offset_bytes;
}

<<<<<<< HEAD
void GpuExecutor::Deallocate(DeviceMemoryBase* mem) {
  GpuDriver::DeviceDeallocate(context_, mem->opaque());
}

bool GpuExecutor::HostMemoryRegister(void* location, uint64 size) {
=======
void CUDAExecutor::Deallocate(DeviceMemoryBase *mem) {
  // CUDA "sub-buffers" are just pointer + offset, so no dealloc is necessary.
  if (!mem->is_sub_buffer()) {
    CUDADriver::DeviceDeallocate(context_, mem->opaque());
  }
}

bool CUDAExecutor::HostMemoryRegister(void *location, uint64 size) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (location == nullptr || size == 0) {
    LOG(WARNING) << "attempting to register null or zero-sized memory: "
                 << location << "; size " << size;
  }
  VLOG(2) << "registering " << location << " size " << size;
<<<<<<< HEAD
  return GpuDriver::HostRegister(context_, location, size);
}

bool GpuExecutor::HostMemoryUnregister(void* location) {
  VLOG(2) << "unregistering " << location;
  return GpuDriver::HostUnregister(context_, location);
}

bool GpuExecutor::SynchronizeAllActivity() {
  return GpuDriver::SynchronizeContext(context_);
}

port::Status GpuExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                             uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsCudaDevicePtr(location), 0x0, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                           0x0, size);
}

port::Status GpuExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                            int value, uint64 size) {
=======
  return CUDADriver::HostRegister(context_, location, size);
}

bool CUDAExecutor::HostMemoryUnregister(void *location) {
  VLOG(2) << "unregistering " << location;
  return CUDADriver::HostUnregister(context_, location);
}

bool CUDAExecutor::SynchronizeAllActivity() {
  return CUDADriver::SynchronizeContext(context_);
}

bool CUDAExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return CUDADriver::SynchronousMemsetUint32(
        context_, AsCudaDevicePtr(location), 0x0, size / 4);
  }
  return CUDADriver::SynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                            0x0, size);
}

bool CUDAExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                     uint64 size) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    // cudaMemset reinterprets "value" as a uint8.
    uint8 byte_value = static_cast<uint8>(value);
    uint32 pattern = (byte_value << 24) | (byte_value << 16) |
                     (byte_value << 8) | byte_value;
<<<<<<< HEAD
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsCudaDevicePtr(location), pattern, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                           value, size);
}

port::Status GpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src, uint64 size) {
  return GpuDriver::SynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                         host_src, size);
}

port::Status GpuExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64 size) {
  return GpuDriver::SynchronousMemcpyD2H(context_, host_dst,
                                         AsCudaDevicePtr(gpu_src), size);
}

port::Status GpuExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64 size) {
  return GpuDriver::SynchronousMemcpyD2D(context_, AsCudaDevicePtr(gpu_dst),
                                         AsCudaDevicePtr(gpu_src), size);
}

port::Status GpuExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                  uint64 size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return Memset32(stream, location, 0x0, size);
  } else {
    return Memset(stream, location, 0x0, size);
  }
}

port::Status GpuExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                 uint8 pattern, uint64 size) {
  VLOG(2) << "enqueueing memset8 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  return GpuDriver::AsynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                            pattern, size,
                                            AsGpuStreamValue(stream));
}

port::Status GpuExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                   uint32 pattern, uint64 size) {
=======
    return CUDADriver::SynchronousMemsetUint32(
        context_, AsCudaDevicePtr(location), pattern, size / 4);
  }
  return CUDADriver::SynchronousMemsetUint8(context_, AsCudaDevicePtr(location),
                                            value, size);
}

bool CUDAExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                     const void *host_src, uint64 size) {
  return CUDADriver::SynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                          host_src, size);
}

bool CUDAExecutor::SynchronousMemcpy(void *host_dst,
                                     const DeviceMemoryBase &gpu_src,
                                     uint64 size) {
  return CUDADriver::SynchronousMemcpyD2H(context_, host_dst,
                                          AsCudaDevicePtr(gpu_src), size);
}

bool CUDAExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src, uint64 size) {
  return CUDADriver::SynchronousMemcpyD2D(context_, AsCudaDevicePtr(gpu_dst),
                                          AsCudaDevicePtr(gpu_src), size);
}

bool CUDAExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                           uint64 size) {
  return Memset32(stream, location, 0x0, size);
}

bool CUDAExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                            uint32 pattern, uint64 size) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  VLOG(2) << "enqueueing memset32 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
        size % 4 == 0);
<<<<<<< HEAD
  return GpuDriver::AsynchronousMemsetUint32(
      context_, AsCudaDevicePtr(location), pattern, size / 4,
      AsGpuStreamValue(stream));
}

bool GpuExecutor::Memcpy(Stream* stream, void* host_dst,
                         const DeviceMemoryBase& gpu_src, uint64 size) {
  return GpuDriver::AsynchronousMemcpyD2H(context_, host_dst,
                                          AsCudaDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                         const void* host_src, uint64 size) {
  return GpuDriver::AsynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                          host_src, size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::MemcpyDeviceToDevice(Stream* stream,
                                       DeviceMemoryBase* gpu_dst,
                                       const DeviceMemoryBase& gpu_src,
                                       uint64 size) {
  return GpuDriver::AsynchronousMemcpyD2D(context_, AsCudaDevicePtr(gpu_dst),
                                          AsCudaDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::HostCallback(Stream* stream,
                               std::function<port::Status()> callback) {
  auto callback_ptr = new std::function<void()>([callback]() {
    port::Status s = callback();
    if (!s.ok()) {
      LOG(WARNING) << "Host callback failed: " << s;
    }
  });
  return GpuDriver::AddStreamCallback(context_, AsGpuStreamValue(stream),
                                      InternalHostCallback, callback_ptr);
}

/* static */ void GpuExecutor::InternalHostCallback(CUstream stream,
                                                    CUresult status,
                                                    void* data) {
=======
  return CUDADriver::AsynchronousMemsetUint32(
      context_, AsCudaDevicePtr(location), pattern, size / 4,
      AsCUDAStreamValue(stream));
}

bool CUDAExecutor::Memcpy(Stream *stream, void *host_dst,
                          const DeviceMemoryBase &gpu_src, uint64 size) {
  return CUDADriver::AsynchronousMemcpyD2H(context_, host_dst,
                                           AsCudaDevicePtr(gpu_src), size,
                                           AsCUDAStreamValue(stream));
}

bool CUDAExecutor::Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                          const void *host_src, uint64 size) {
  return CUDADriver::AsynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                                           host_src, size,
                                           AsCUDAStreamValue(stream));
}

bool CUDAExecutor::MemcpyDeviceToDevice(Stream *stream,
                                        DeviceMemoryBase *gpu_dst,
                                        const DeviceMemoryBase &gpu_src,
                                        uint64 size) {
  return CUDADriver::AsynchronousMemcpyD2D(context_, AsCudaDevicePtr(gpu_dst),
                                           AsCudaDevicePtr(gpu_src), size,
                                           AsCUDAStreamValue(stream));
}

bool CUDAExecutor::HostCallback(Stream *stream,
                                std::function<void()> callback) {
  auto callback_ptr = new std::function<void()>(callback);
  return CUDADriver::AddStreamCallback(context_, AsCUDAStreamValue(stream),
                                       InternalHostCallback, callback_ptr);
}

/* static */ void CUDAExecutor::InternalHostCallback(CUstream stream,
                                                     CUresult status,
                                                     void *data) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  std::function<void()> *callback =
      reinterpret_cast<std::function<void()> *>(data);
  (*callback)();
  delete callback;
}

<<<<<<< HEAD
port::Status GpuExecutor::AllocateEvent(Event* event) {
  return AsGpuEvent(event)->Init();
}

port::Status GpuExecutor::DeallocateEvent(Event* event) {
  return AsGpuEvent(event)->Destroy();
}

port::Status GpuExecutor::RecordEvent(Stream* stream, Event* event) {
  return AsGpuEvent(event)->Record(AsGpuStream(stream));
}

port::Status GpuExecutor::WaitForEvent(Stream* stream, Event* event) {
  if (GpuDriver::WaitStreamOnEvent(context_, AsGpuStream(stream)->gpu_stream(),
                                   AsGpuEvent(event)->gpu_event())) {
    return port::Status::OK();
  } else {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat("error recording waiting for CUDA event on stream %p",
                        stream));
  }
}

Event::Status GpuExecutor::PollForEventStatus(Event* event) {
  return AsGpuEvent(event)->PollForStatus();
}

bool GpuExecutor::AllocateStream(Stream* stream) {
  return AsGpuStream(stream)->Init();
}

void GpuExecutor::DeallocateStream(Stream* stream) {
  GpuStream* cuda_stream = AsGpuStream(stream);
=======
port::Status CUDAExecutor::AllocateEvent(Event *event) {
  return AsCUDAEvent(event)->Init();
}

port::Status CUDAExecutor::DeallocateEvent(Event *event) {
  return AsCUDAEvent(event)->Destroy();
}

port::Status CUDAExecutor::RecordEvent(Stream *stream, Event *event) {
  return AsCUDAEvent(event)->Record(AsCUDAStream(stream));
}

port::Status CUDAExecutor::WaitForEvent(Stream *stream, Event *event) {
  if (CUDADriver::WaitStreamOnEvent(context_,
                                    AsCUDAStream(stream)->cuda_stream(),
                                    AsCUDAEvent(event)->cuda_event())) {
    return port::Status::OK();
  } else {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("error recording waiting for CUDA event on stream %p",
                     stream)};
  }
}

Event::Status CUDAExecutor::PollForEventStatus(Event *event) {
  return AsCUDAEvent(event)->PollForStatus();
}

bool CUDAExecutor::AllocateStream(Stream *stream) {
  return AsCUDAStream(stream)->Init();
}

void CUDAExecutor::DeallocateStream(Stream *stream) {
  CUDAStream *cuda_stream = AsCUDAStream(stream);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!cuda_stream->IsIdle()) {
    LOG(ERROR) << "Deallocating stream with pending work";
  }
  cuda_stream->Destroy();
}

<<<<<<< HEAD
bool GpuExecutor::AllocateTimer(Timer* timer) {
  return AsGpuTimer(timer)->Init();
}

void GpuExecutor::DeallocateTimer(Timer* timer) {
  AsGpuTimer(timer)->Destroy();
}

bool GpuExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  CUevent other_completed_event = *AsGpuStream(other)->completed_event();
  bool ok = GpuDriver::RecordEvent(context_, other_completed_event,
                                   AsGpuStreamValue(other))
                .ok();
=======
bool CUDAExecutor::AllocateTimer(Timer *timer) {
  return AsCUDATimer(timer)->Init();
}

void CUDAExecutor::DeallocateTimer(Timer *timer) {
  AsCUDATimer(timer)->Destroy();
}

bool CUDAExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  CUevent other_completed_event;
  bool ok =
      AsCUDAStream(other)->GetOrCreateCompletedEvent(&other_completed_event);
  if (!ok) {
    LOG(ERROR) << "failed to get completion event from other; "
                  "therefore, failed to create inter-stream dependency";
    return false;
  }

  ok = CUDADriver::RecordEvent(context_, other_completed_event,
                               AsCUDAStreamValue(other))
           .ok();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!ok) {
    LOG(ERROR) << "failed to record completion event; "
                  "therefore, failed to create inter-stream dependency";
    return false;
  }

<<<<<<< HEAD
  return GpuDriver::WaitStreamOnEvent(context_, AsGpuStreamValue(dependent),
                                      other_completed_event);
}

bool GpuExecutor::StartTimer(Stream* stream, Timer* timer) {
  return AsGpuTimer(timer)->Start(AsGpuStream(stream));
}

bool GpuExecutor::StopTimer(Stream* stream, Timer* timer) {
  return AsGpuTimer(timer)->Stop(AsGpuStream(stream));
}

port::Status GpuExecutor::BlockHostUntilDone(Stream* stream) {
  return GpuDriver::SynchronizeStream(context_, AsGpuStreamValue(stream));
}

blas::BlasSupport* GpuExecutor::CreateBlas() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(cuda::kCudaPlatformId,
=======
  return CUDADriver::WaitStreamOnEvent(context_, AsCUDAStreamValue(dependent),
                                       other_completed_event);
}

bool CUDAExecutor::StartTimer(Stream *stream, Timer *timer) {
  return AsCUDATimer(timer)->Start(AsCUDAStream(stream));
}

bool CUDAExecutor::StopTimer(Stream *stream, Timer *timer) {
  return AsCUDATimer(timer)->Stop(AsCUDAStream(stream));
}

bool CUDAExecutor::BlockHostUntilDone(Stream *stream) {
  return CUDADriver::SynchronizeStream(context_, AsCUDAStreamValue(stream));
}

blas::BlasSupport *CUDAExecutor::CreateBlas() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(kCudaPlatformId,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

<<<<<<< HEAD
dnn::DnnSupport* GpuExecutor::CreateDnn() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(cuda::kCudaPlatformId,
=======
dnn::DnnSupport *CUDAExecutor::CreateDnn() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(kCudaPlatformId,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                                       plugin_config_.dnn());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

<<<<<<< HEAD
fft::FftSupport* GpuExecutor::CreateFft() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(cuda::kCudaPlatformId,
=======
fft::FftSupport *CUDAExecutor::CreateFft() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(kCudaPlatformId,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

<<<<<<< HEAD
rng::RngSupport* GpuExecutor::CreateRng() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(cuda::kCudaPlatformId,
=======
rng::RngSupport *CUDAExecutor::CreateRng() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(kCudaPlatformId,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

// TODO(rspringer): Remove in b/18544742.
<<<<<<< HEAD
bool GpuExecutor::SupportsDnn() const { return true; }

bool GpuExecutor::CanEnablePeerAccessTo(StreamExecutorInterface* other) {
  GpuExecutor* cuda_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::CanEnablePeerAccess(context_, cuda_other->context_);
}

port::Status GpuExecutor::EnablePeerAccessTo(StreamExecutorInterface* other) {
  GpuExecutor* cuda_other = static_cast<GpuExecutor*>(other);
  return GpuDriver::EnablePeerAccess(context_, cuda_other->context_);
}

SharedMemoryConfig GpuExecutor::GetDeviceSharedMemoryConfig() {
  port::StatusOr<CUsharedconfig> cuda_config =
      GpuDriver::ContextGetSharedMemConfig(context_);
=======
bool CUDAExecutor::SupportsDnn() const {
  return true;
}

bool CUDAExecutor::CanEnablePeerAccessTo(StreamExecutorInterface *other) {
  CUDAExecutor *cuda_other = static_cast<CUDAExecutor *>(other);
  return CUDADriver::CanEnablePeerAccess(context_, cuda_other->context_);
}

port::Status CUDAExecutor::EnablePeerAccessTo(StreamExecutorInterface *other) {
  CUDAExecutor *cuda_other = static_cast<CUDAExecutor *>(other);
  return CUDADriver::EnablePeerAccess(context_, cuda_other->context_);
}

SharedMemoryConfig CUDAExecutor::GetDeviceSharedMemoryConfig() {
  port::StatusOr<CUsharedconfig> cuda_config =
      CUDADriver::ContextGetSharedMemConfig(context_);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!cuda_config.ok()) {
    // Don't log; the failed call will log necessary output.
    return SharedMemoryConfig::kDefault;
  }

  switch (cuda_config.ValueOrDie()) {
    case CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE:
      return SharedMemoryConfig::kDefault;
    case CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE:
      return SharedMemoryConfig::kFourByte;
    case CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE:
      return SharedMemoryConfig::kEightByte;
    default:
      LOG(FATAL) << "Invalid shared memory configuration returned: "
                 << cuda_config.ValueOrDie();
  }
}

<<<<<<< HEAD
port::Status GpuExecutor::SetDeviceSharedMemoryConfig(
=======
port::Status CUDAExecutor::SetDeviceSharedMemoryConfig(
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    SharedMemoryConfig config) {
  CUsharedconfig cuda_config;
  switch (config) {
    case SharedMemoryConfig::kDefault:
      cuda_config = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
      break;
    case SharedMemoryConfig::kFourByte:
      cuda_config = CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE;
      break;
    case SharedMemoryConfig::kEightByte:
      cuda_config = CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE;
      break;
    default:
      LOG(FATAL) << "Invalid shared memory configuration specified: "
                 << static_cast<int>(config);
  }
<<<<<<< HEAD
  return GpuDriver::ContextSetSharedMemConfig(context_, cuda_config);
}

bool GpuExecutor::DeviceMemoryUsage(int64* free, int64* total) const {
  return GpuDriver::GetDeviceMemoryInfo(context_, free, total);
}

bool GpuExecutor::GetSymbol(const string& symbol_name,
                            ModuleHandle module_handle, void** mem,
                            size_t* bytes) {
  auto lookup_in_module = [&](CUmodule module) {
    CHECK(module != nullptr);
    return GpuDriver::GetModuleSymbol(context_, module, symbol_name.c_str(),
                                      reinterpret_cast<CUdeviceptr*>(mem),
                                      bytes);
  };

  {  // give limited scope to mutex_lock
    absl::MutexLock lock{&in_memory_modules_mu_};
    if (static_cast<bool>(module_handle)) {
      auto it = gpu_binary_to_module_.find(module_handle.id());
      CHECK(it != gpu_binary_to_module_.end());
      return lookup_in_module(it->second.first);
    }

    for (auto &it : gpu_binary_to_module_) {
      if (lookup_in_module(it.second.first)) {
=======
  return CUDADriver::ContextSetSharedMemConfig(context_, cuda_config);
}

bool CUDAExecutor::DeviceMemoryUsage(int64 *free, int64 *total) const {
  return CUDADriver::GetDeviceMemoryInfo(context_, free, total);
}

bool CUDAExecutor::GetSymbol(const string& symbol_name, void **mem,
                             size_t *bytes) {
  {  // give limited scope to mutex_lock
    mutex_lock lock{disk_modules_mu_};
    for (auto &it : disk_modules_) {
      if (CUDADriver::GetModuleSymbol(context_, it.second, symbol_name.c_str(),
                                      reinterpret_cast<CUdeviceptr *>(mem),
                                      bytes)) {
        return true;
      }
    }
  }

  {  // give limited scope to mutex_lock
    mutex_lock lock{in_memory_modules_mu_};
    for (auto &it : in_memory_modules_) {
      if (CUDADriver::GetModuleSymbol(context_, it.second, symbol_name.c_str(),
                                      reinterpret_cast<CUdeviceptr *>(mem),
                                      bytes)) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
        return true;
      }
    }
  }

<<<<<<< HEAD
  LOG(INFO) << "Failed to find symbol in any modules: " << symbol_name;
  return false;
}

bool FillBlockDimLimit(GpuDeviceHandle device, BlockDim* block_dim_limit) {
=======
  LOG(INFO) << "Falied to find symbol in any modules: " << symbol_name;
  return false;
}

bool CUDAExecutor::FillBlockDimLimit(BlockDim *block_dim_limit) const {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
<<<<<<< HEAD
  if (!GpuDriver::GetGridLimits(&x, &y, &z, device)) {
=======
  if (!CUDADriver::GetGridLimits(&x, &y, &z, device_)) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    return false;
  }

  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return true;
}

<<<<<<< HEAD
bool GpuExecutor::SupportsBlas() const { return true; }

bool GpuExecutor::SupportsFft() const { return true; }

bool GpuExecutor::SupportsRng() const { return true; }

std::unique_ptr<internal::EventInterface>
GpuExecutor::CreateEventImplementation() {
  return std::unique_ptr<internal::EventInterface>(new GpuEvent(this));
}

std::unique_ptr<internal::KernelInterface>
GpuExecutor::CreateKernelImplementation() {
  return std::unique_ptr<internal::KernelInterface>(new GpuKernel());
}

std::unique_ptr<internal::StreamInterface>
GpuExecutor::GetStreamImplementation() {
  return std::unique_ptr<internal::StreamInterface>(new GpuStream(this));
}

std::unique_ptr<internal::TimerInterface>
GpuExecutor::GetTimerImplementation() {
  return std::unique_ptr<internal::TimerInterface>(new GpuTimer(this));
}

void* GpuExecutor::GpuContextHack() { return context_; }

GpuContext* GpuExecutor::gpu_context() { return context_; }

// Attempts to read the NUMA node corresponding to the GPU device's PCI bus out
=======
KernelArg CUDAExecutor::DeviceMemoryToKernelArg(
    const DeviceMemoryBase &gpu_mem) const {
  const void* arg = gpu_mem.opaque();
  const uint8 *arg_ptr = reinterpret_cast<const uint8 *>(&arg);

  KernelArg kernel_arg;
  kernel_arg.type = KernelArg::kNormal;
  kernel_arg.data = port::InlinedVector<uint8, 4>(arg_ptr, arg_ptr + sizeof(arg));
  kernel_arg.bytes = sizeof(arg);
  return kernel_arg;
}

bool CUDAExecutor::SupportsBlas() const { return true; }

bool CUDAExecutor::SupportsFft() const { return true; }

bool CUDAExecutor::SupportsRng() const { return true; }

void *CUDAExecutor::CudaContextHack() { return context_; }

CUcontext CUDAExecutor::cuda_context() { return context_; }

// Attemps to read the NUMA node corresponding to the GPU device's PCI bus out
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// of SysFS. Returns -1 if it cannot.
//
// For anything more complicated/prod-focused than this, you'll likely want to
// turn to gsys' topology modeling.
static int TryToReadNumaNode(const string &pci_bus_id, int device_ordinal) {
<<<<<<< HEAD
#if defined(__APPLE__)
  LOG(INFO) << "OS X does not support NUMA - returning NUMA node zero";
  return 0;
#elif defined(PLATFORM_WINDOWS)
  // Windows support for NUMA is not currently implemented. Return node 0.
  return 0;
#elif defined(__aarch64__)
  LOG(INFO) << "ARM64 does not support NUMA - returning NUMA node zero";
  return 0;
#else
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  VLOG(2) << "trying to read NUMA node for device ordinal: " << device_ordinal;
  static const int kUnknownNumaNode = -1;

  if (pci_bus_id.empty()) {
    LOG(INFO) << "no PCI bus ID for device ordinal: " << device_ordinal;
    return kUnknownNumaNode;
  }

  string filename =
<<<<<<< HEAD
      absl::StrFormat("/sys/bus/pci/devices/%s/numa_node", pci_bus_id);
=======
      port::Printf("/sys/bus/pci/devices/%s/numa_node", pci_bus_id.c_str());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // We have to use fopen/fread here so that the device properties can be
  // populated before InitGoogle procedure has been completed (at which point we
  // could use the file::* utilities).
  FILE *file = fopen(filename.c_str(), "r");
  if (file == nullptr) {
<<<<<<< HEAD
    LOG(ERROR) << "could not open file to read NUMA node: " << filename
               << "\nYour kernel may have been built without NUMA support.";
=======
    LOG(ERROR) << "could not open file to read NUMA node: " << filename;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    return kUnknownNumaNode;
  }

  string content;
  char buf[32];
  size_t did_read = fread(buf, sizeof(buf[0]), sizeof(buf) - 1, file);
  buf[did_read] = '\0';
  content = buf;

  int32 value;
  if (port::safe_strto32(content, &value)) {
    if (value < 0) {  // See http://b/18228951 for details on this path.
      LOG(INFO) << "successful NUMA node read from SysFS had negative value ("
                << value << "), but there must be at least one NUMA node"
                            ", so returning NUMA node zero";
<<<<<<< HEAD
      fclose(file);
      return 0;
    }
    fclose(file);
=======
      return 0;
    }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    return value;
  }

  LOG(WARNING)
      << "could not convert SysFS file contents to integral NUMA node value: "
      << content;

<<<<<<< HEAD
  fclose(file);
  return kUnknownNumaNode;
#endif
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
GpuExecutor::CreateDeviceDescription(int device_ordinal) {
  GpuDeviceHandle device;
  auto status = GpuDriver::GetDevice(device_ordinal, &device);
  if (!status.ok()) {
    return status;
  }

  int cc_major;
  int cc_minor;
  status = GpuDriver::GetComputeCapability(&cc_major, &cc_minor, device);
  if (!status.ok()) {
    return status;
  }

=======
  return kUnknownNumaNode;
}

// Set of compute capability specific device parameters that cannot be
// queried from the driver API.  These values instead are baked into a
// lookup table indexed by compute capability version.
struct UnqueryableDeviceParams {
  int cc_major;
  int cc_minor;
  uint64 blocks_per_core_limit;
  uint64 registers_per_core_limit;
  uint64 registers_per_thread_limit;
  uint64 warp_alloc_granularity;
  uint64 register_alloc_granularity;
  uint64 shared_memory_alloc_granularity;
};

static const UnqueryableDeviceParams kAllUnqueryableDeviceParams[] = {
  {
    3, 5,       // compute capability (3.5)
    16,         // blocks_per_core_limit
    64 * 1024,  // registers_per_core_limit
    255,        // registers_per_thread_limit
    4,          // warp_alloc_granularity
    256,        // register_alloc_granularity
    256         // shared_memory_alloc_granularity
  }
};

DeviceDescription *CUDAExecutor::PopulateDeviceDescription() const {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  internal::DeviceDescriptionBuilder builder;

  {
    int driver_version = 0;
<<<<<<< HEAD
    (void)GpuDriver::GetDriverVersion(&driver_version);
    string augmented_driver_version = absl::StrFormat(
        "%d (%s)", driver_version,
        cuda::DriverVersionStatusToString(Diagnostician::FindDsoVersion()));
=======
    (void)CUDADriver::GetDriverVersion(&driver_version);
    string augmented_driver_version = port::Printf(
        "%d (%s)", driver_version,
        DriverVersionStatusToString(Diagnostician::FindDsoVersion()).c_str());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    builder.set_driver_version(augmented_driver_version);
  }

  {
<<<<<<< HEAD
    string pci_bus_id = GpuDriver::GetPCIBusID(device);

    // Lower the hex characters to match sysfs.
    pci_bus_id = absl::AsciiStrToLower(pci_bus_id);
    builder.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = TryToReadNumaNode(pci_bus_id, device_ordinal);
    builder.set_numa_node(numa_node);
  }

  {
    builder.set_threads_per_block_limit(
        GpuDriver::GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                      device)
            .ValueOrDie());

    ThreadDim thread_dim_limit;
    thread_dim_limit.x = GpuDriver::GetDeviceAttribute(
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device)
                             .ValueOrDie();
    thread_dim_limit.y = GpuDriver::GetDeviceAttribute(
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device)
                             .ValueOrDie();
    thread_dim_limit.z = GpuDriver::GetDeviceAttribute(
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device)
                             .ValueOrDie();
    builder.set_thread_dim_limit(thread_dim_limit);

    int clock_rate =
        GpuDriver::GetDeviceAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device)
            .ValueOrDie();
    builder.set_clock_rate_ghz(static_cast<float>(clock_rate) / 1e6);
=======
    string pci_bus_id = CUDADriver::GetPCIBusID(device_);

    // Lower the hex characters to match sysfs.
    pci_bus_id = port::Lowercase(pci_bus_id);
    builder.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    int numa_node = TryToReadNumaNode(pci_bus_id, device_ordinal_);
    builder.set_numa_node(numa_node);
  }

  CUdevprop prop;
  if (CUDADriver::GetDeviceProperties(&prop, device_ordinal_)) {
    builder.set_threads_per_block_limit(prop.maxThreadsPerBlock);

    ThreadDim thread_dim_limit;
    thread_dim_limit.x = prop.maxThreadsDim[0];
    thread_dim_limit.y = prop.maxThreadsDim[1];
    thread_dim_limit.z = prop.maxThreadsDim[2];
    builder.set_thread_dim_limit(thread_dim_limit);

    float clock_rate_ghz = static_cast<float>(prop.clockRate) / 1e6;
    builder.set_clock_rate_ghz(clock_rate_ghz);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  {
    bool ecc_enabled = false;
<<<<<<< HEAD
    (void)GpuDriver::IsEccEnabled(device, &ecc_enabled);
=======
    (void)CUDADriver::IsEccEnabled(device_, &ecc_enabled);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    builder.set_ecc_enabled(ecc_enabled);
  }

  {
    uint64 device_memory_size = -1;
<<<<<<< HEAD
    (void)GpuDriver::GetDeviceTotalMemory(device, &device_memory_size);
    builder.set_device_memory_size(device_memory_size);
  }

  port::StatusOr<int> mem_clock_khz = GpuDriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device_ordinal);
  port::StatusOr<int> mem_bus_width_bits = GpuDriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device_ordinal);
  if (mem_clock_khz.ok() && mem_bus_width_bits.ok()) {
    // Times 2 because HBM is DDR memory; it gets two data bits per each data
    // lane.
    builder.set_memory_bandwidth(2 * int64_t{mem_clock_khz.ValueOrDie()} *
                                 1000 *
                                 int64_t{mem_bus_width_bits.ValueOrDie()} / 8);
  }

  {
    BlockDim block_dim_limit;
    FillBlockDimLimit(device, &block_dim_limit);
=======
    (void)CUDADriver::GetDeviceTotalMemory(device_, &device_memory_size);
    builder.set_device_memory_size(device_memory_size);
  }

  {
    BlockDim block_dim_limit;
    FillBlockDimLimit(&block_dim_limit);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    string device_name;
<<<<<<< HEAD
    TF_RETURN_IF_ERROR(GpuDriver::GetDeviceName(device, &device_name));
    builder.set_name(device_name);
  }

  builder.set_platform_version(
      absl::StrCat("Compute Capability ", cc_major, ".", cc_minor));
=======
    (void)CUDADriver::GetDeviceName(device_, &device_name);
    builder.set_name(device_name);
  }

  for (size_t i = 0; i < ARRAYSIZE(kAllUnqueryableDeviceParams); i++) {
    const auto &params = kAllUnqueryableDeviceParams[i];
    if (params.cc_major == cc_major_ && params.cc_minor == cc_minor_) {
      builder.set_blocks_per_core_limit(params.blocks_per_core_limit);
      builder.set_registers_per_core_limit(params.registers_per_core_limit);
      builder.set_registers_per_thread_limit(params.registers_per_thread_limit);
      builder.set_warp_alloc_granularity(params.warp_alloc_granularity);
      builder.set_register_alloc_granularity(params.register_alloc_granularity);
      builder.set_shared_memory_alloc_granularity(
          params.shared_memory_alloc_granularity);
    }
  }

  builder.set_platform_version(
      port::StrCat("Compute Capability ", cc_major_, ".", cc_minor_));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  builder.set_device_address_bits(64);

  builder.set_device_vendor("NVIDIA Corporation");
<<<<<<< HEAD
  builder.set_cuda_compute_capability(cc_major, cc_minor);
  builder.set_shared_memory_per_core(
      GpuDriver::GetMaxSharedMemoryPerCore(device).ValueOrDie());
  builder.set_shared_memory_per_block(
      GpuDriver::GetMaxSharedMemoryPerBlock(device).ValueOrDie());
  builder.set_core_count(
      GpuDriver::GetMultiprocessorCount(device).ValueOrDie());
  builder.set_threads_per_core_limit(
      GpuDriver::GetMaxThreadsPerMultiprocessor(device).ValueOrDie());
  builder.set_registers_per_block_limit(
      GpuDriver::GetMaxRegistersPerBlock(device).ValueOrDie());
  builder.set_threads_per_warp(
      GpuDriver::GetThreadsPerWarp(device).ValueOrDie());
  builder.set_registers_per_core_limit(
      GpuDriver::GetDeviceAttribute(
          CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device)
          .ValueOrDie());

  return builder.Build();
}

}  // namespace gpu

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(cuda_gpu_executor, {});
=======
  builder.set_cuda_compute_capability(cc_major_, cc_minor_);
  builder.set_shared_memory_per_core(
      CUDADriver::GetMaxSharedMemoryPerCore(device_).ValueOrDie());
  builder.set_shared_memory_per_block(
      CUDADriver::GetMaxSharedMemoryPerBlock(device_).ValueOrDie());
  builder.set_core_count(
      CUDADriver::GetMultiprocessorCount(device_).ValueOrDie());
  builder.set_threads_per_core_limit(
      CUDADriver::GetMaxThreadsPerMultiprocessor(device_).ValueOrDie());
  builder.set_registers_per_block_limit(
      CUDADriver::GetMaxRegistersPerBlock(device_).ValueOrDie());
  builder.set_threads_per_warp(
      CUDADriver::GetThreadsPerWarp(device_).ValueOrDie());

  auto built = builder.Build();
  return built.release();
}

}  // namespace cuda

namespace gpu = ::perftools::gputools;

void initialize_cuda_gpu_executor() {
  port::StatusOr<void *> status =
      gpu::internal::CachedDsoLoader::GetLibcudaDsoHandle();
  if (!status.ok()) {
    gpu::cuda::Diagnostician::LogDriverVersionInformation();
    LOG(INFO) << "LD_LIBRARY_PATH: " << getenv("LD_LIBRARY_PATH");
    LOG(INFO) << "failed to find libcuda.so on this system: "
              << status.status();
  }

  // TODO(b/22689637): Temporary until users are migrated off of PlatformKind.
  gpu::PluginRegistry::Instance()->MapPlatformKindToId(
      gpu::PlatformKind::kCuda, gpu::cuda::kCudaPlatformId);

  *gpu::internal::MakeCUDAExecutorImplementation() = [](
      const gpu::PluginConfig &config) {
    return new gpu::cuda::CUDAExecutor{config};
  };

  *gpu::internal::MakeCUDAKernelImplementation() = []() {
    return new gpu::cuda::CUDAKernel;
  };

  *gpu::internal::MakeCUDAEventImplementation() = [](
      gpu::StreamExecutor *parent) {
    gpu::cuda::CUDAExecutor *cuda_executor =
        static_cast<gpu::cuda::CUDAExecutor *>(parent->implementation());
    return new gpu::cuda::CUDAEvent{cuda_executor};
  };

  *gpu::internal::MakeCUDAStreamImplementation() = [](
      gpu::StreamExecutor *parent) {
    gpu::cuda::CUDAExecutor *cuda_executor =
        static_cast<gpu::cuda::CUDAExecutor *>(parent->implementation());
    return new gpu::cuda::CUDAStream{cuda_executor};
  };
  *gpu::internal::MakeCUDATimerImplementation() = [](
      gpu::StreamExecutor *parent) {
    gpu::cuda::CUDAExecutor *cuda_executor =
        static_cast<gpu::cuda::CUDAExecutor *>(parent->implementation());
    return new gpu::cuda::CUDATimer{cuda_executor};
  };
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(
    cuda_gpu_executor, {perftools::gputools::initialize_cuda_gpu_executor();});
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
