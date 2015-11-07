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
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"

#include <unordered_set>

<<<<<<< HEAD
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/allocator.h"
=======
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#if (!defined(PLATFORM_POSIX_ANDROID) && !defined(PLATFORM_GOOGLE_ANDROID)) && \
    (defined(PLATFORM_GOOGLE) || GOOGLE_CUDA)
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
<<<<<<< HEAD
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
=======
#include "tensorflow/core/platform/port.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

namespace {
<<<<<<< HEAD
void SameWorkerRecvDone(const DeviceMgr* device_mgr,
                        const Rendezvous::ParsedKey& parsed,
                        const Rendezvous::Args& send_args,
                        const Rendezvous::Args& recv_args, const Tensor& in,
                        Tensor* out, StatusCallback done) {
=======

void CopyTensorBetweenDevices(const string& id, DeviceContext* send_dev_context,
                              DeviceContext* recv_dev_context, Device* src,
                              Device* dst,
                              const AllocatorAttributes src_alloc_attr,
                              const AllocatorAttributes dst_alloc_attr,
                              const Tensor* input, Tensor* output,
                              std::function<void(const Status&)> done) {
  if (src->attributes().device_type() != dst->attributes().device_type()) {
    done(errors::Unimplemented(
        "Copy between device types not yet implemented: src=", src->name(),
        " dst=", dst->name()));
  } else if (src->attributes().device_type() != "CPU") {
    done(errors::Unimplemented(
        "Copy between non-CPU devices not yet implemented"));
  }
  *output = *input;
  done(Status::OK());
}

#if (!defined(PLATFORM_POSIX_ANDROID) && !defined(PLATFORM_GOOGLE_ANDROID)) && \
    (defined(PLATFORM_GOOGLE) || GOOGLE_CUDA)
constexpr auto CopyTensorBetweenDevicesFunc = &GPUUtil::CopyViaDMA;
#else
constexpr auto CopyTensorBetweenDevicesFunc = &CopyTensorBetweenDevices;
#endif

}  // end namespace

IntraProcessRendezvous::IntraProcessRendezvous(const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr), local_(NewLocalRendezvous()) {}

IntraProcessRendezvous::~IntraProcessRendezvous() { local_->Unref(); }

Status IntraProcessRendezvous::Send(const string& key,
                                    const Rendezvous::Args& args,
                                    const Tensor& val, const bool is_dead) {
  VLOG(1) << "IntraProcessRendezvous Send " << this << " " << key;
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
  }
  Rendezvous::ParsedKey parsed;
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(key, &parsed));

  // Buffers "val" and "device_context" in local_.
  return local_->Send(key, args, val, is_dead);
}

Status IntraProcessRendezvous::ParseKey(const string& key, bool is_src,
                                        Rendezvous::ParsedKey* parsed) {
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
  }
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(key, parsed));
  return Status::OK();
}

void IntraProcessRendezvous::SameWorkerRecvDone(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& send_args,
    const Rendezvous::Args& recv_args, const Tensor& in, Tensor* out,
    StatusCallback done) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // Do a quick copy (sharing the underlying buffer) if both tensors
  // are on host memory.
  const bool src_host =
      (send_args.alloc_attrs.on_host() || parsed.src.type == "CPU");
  const bool dst_host =
      (recv_args.alloc_attrs.on_host() || parsed.dst.type == "CPU");
  if (src_host && dst_host) {
    *out = in;
    done(Status::OK());
    return;
  }

  // This copy must involve a non-CPU device. Hence, "in" must support DMA
<<<<<<< HEAD
  // (e.g., string tensors do not work on GPU).  Variant copy DMA
  // checks happen inside CopyTensor::ViaDMA.
  if (!DataTypeCanUseMemcpy(in.dtype()) && in.dtype() != DT_VARIANT &&
      in.dtype() != DT_RESOURCE) {
    done(errors::InvalidArgument(
        "Non-DMA-safe ", DataTypeString(in.dtype()),
        " tensor may not be copied from/to a device. Key: ", parsed.FullKey()));
=======
  // (e.g., string tensors do not work on GPU).
  if (!DataTypeCanUseMemcpy(in.dtype())) {
    done(errors::InvalidArgument("Non-DMA-safe ", DataTypeString(in.dtype()),
                                 " tensor may not be copied from/to a GPU."));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    return;
  }

  Device* src_device;
<<<<<<< HEAD
  Status s = device_mgr->LookupDevice(parsed.src_device, &src_device);
=======
  Status s = device_mgr_->LookupDevice(parsed.src_device, &src_device);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!s.ok()) {
    done(s);
    return;
  }
  Device* dst_device;
<<<<<<< HEAD
  s = device_mgr->LookupDevice(parsed.dst_device, &dst_device);
=======
  s = device_mgr_->LookupDevice(parsed.dst_device, &dst_device);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!s.ok()) {
    done(s);
    return;
  }

<<<<<<< HEAD
  MEMDEBUG_CACHE_OP("SameWorkerRecvDone");
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  AllocatorAttributes attr = recv_args.alloc_attrs;
  attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() ||
                          recv_args.alloc_attrs.gpu_compatible());
  Allocator* out_allocator = dst_device->GetAllocator(attr);
<<<<<<< HEAD
  bool sync_dst_compute = true;
  if (in.dtype() != DT_VARIANT) {
    // Variants are handled by CopyTensor::ViaDMA.
    AllocationAttributes aa;
    uint64 safe_alloc_frontier = dst_device->SafeAllocFrontier(0);
    std::function<uint64()> freed_by_func = [dst_device,
                                             &safe_alloc_frontier]() {
      safe_alloc_frontier = dst_device->SafeAllocFrontier(safe_alloc_frontier);
      return safe_alloc_frontier;
    };
    if (parsed.dst.type == "GPU" && safe_alloc_frontier > 0) {
      // There's a timestamped allocator at work, so use it instead
      // of sync_dst_compute.
      aa.freed_by_func = &freed_by_func;
      sync_dst_compute = false;
    }
    Tensor copy(out_allocator, in.dtype(), in.shape(), aa);
    *out = copy;
  }

  CopyTensor::ViaDMA(
      parsed.edge_name, send_args.device_context, recv_args.device_context,
      src_device, dst_device, send_args.alloc_attrs, recv_args.alloc_attrs, &in,
      out, 0 /*dev_to_dev_stream_index*/, std::move(done), sync_dst_compute);
}

void IntraProcessRecvAsyncImpl(const DeviceMgr* device_mgr,
                               LocalRendezvous* local,
                               const RendezvousInterface::ParsedKey& parsed,
                               const Rendezvous::Args& recv_args,
                               RendezvousInterface::DoneCallback done) {
  VLOG(1) << "IntraProcessRendezvous Recv " << local << " " << parsed.FullKey();

  MEMDEBUG_CACHE_OP("RecvAsync");
  // Recv the tensor from local_.
  local->RecvAsync(
      parsed, recv_args,
      [device_mgr, parsed, done = std::move(done)](
          const Status& status, const Rendezvous::Args& send_args,
          const Rendezvous::Args& recv_args, const Tensor& in,
          bool is_dead) mutable {
        // If "in" is an uninitialized tensor, do copy-construction to
        // preserve the uninitialized state, along with data type and shape
        // info, which is useful for debugger purposes.
        Tensor* out = in.IsInitialized() ? new Tensor : new Tensor(in);

        auto final_callback = [send_args, recv_args, out, is_dead,
                               done = std::move(done)](const Status& s) {
          done(s, send_args, recv_args, *out, is_dead);
          delete out;
        };

        if (status.ok() && in.IsInitialized()) {
          SameWorkerRecvDone(device_mgr, parsed, send_args, recv_args, in, out,
                             std::move(final_callback));
        } else {
          final_callback(status);
        }
      });
}

}  // namespace

RefCountedIntraProcessRendezvous::RefCountedIntraProcessRendezvous(
    const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr) {}

RefCountedIntraProcessRendezvous::~RefCountedIntraProcessRendezvous() {}

Status RefCountedIntraProcessRendezvous::Send(const ParsedKey& key,
                                              const Rendezvous::Args& args,
                                              const Tensor& val,
                                              const bool is_dead) {
  VLOG(1) << "IntraProcessRendezvous Send " << this << " " << key.FullKey();
  return local_.Send(key, args, val, is_dead);
}

void RefCountedIntraProcessRendezvous::RecvAsync(const ParsedKey& key,
                                                 const Rendezvous::Args& args,
                                                 DoneCallback done) {
  VLOG(1) << "IntraProcessRendezvous Recv " << this << " " << key.FullKey();
  IntraProcessRecvAsyncImpl(device_mgr_, &local_, key, args, std::move(done));
}

void RefCountedIntraProcessRendezvous::StartAbort(const Status& s) {
  local_.StartAbort(s);
}

PrivateIntraProcessRendezvous::PrivateIntraProcessRendezvous(
    const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr) {}

PrivateIntraProcessRendezvous::~PrivateIntraProcessRendezvous() {}

Status PrivateIntraProcessRendezvous::Send(const ParsedKey& key,
                                           const Rendezvous::Args& args,
                                           const Tensor& val,
                                           const bool is_dead) {
  DVLOG(1) << "IntraProcessRendezvous Send " << this << " " << key.FullKey();
  return local_.Send(key, args, val, is_dead);
}

void PrivateIntraProcessRendezvous::RecvAsync(const ParsedKey& key,
                                              const Rendezvous::Args& args,
                                              DoneCallback done) {
  DVLOG(1) << "StackAllocatedIntraProcessRendezvous Recv " << this << " "
           << key.FullKey();
  IntraProcessRecvAsyncImpl(device_mgr_, &local_, key, args, std::move(done));
}

void PrivateIntraProcessRendezvous::StartAbort(const Status& s) {
  local_.StartAbort(s);
=======
  Tensor copy(out_allocator, in.dtype(), in.shape());
  *out = copy;

  CopyTensorBetweenDevicesFunc(parsed.edge_name, send_args.device_context,
                               recv_args.device_context, src_device, dst_device,
                               send_args.alloc_attrs, recv_args.alloc_attrs,
                               &in, out, done);
}

void IntraProcessRendezvous::RecvAsync(const string& key,
                                       const Rendezvous::Args& recv_args,
                                       DoneCallback done) {
  VLOG(1) << "IntraProcessRendezvous Recv " << this << " " << key;

  Rendezvous::ParsedKey parsed;
  Status s = ParseKey(key, false /*!is_src*/, &parsed);
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }

  // Recv the tensor from local_.
  local_->RecvAsync(key, recv_args, [this, parsed, done](
                                        const Status& status,
                                        const Rendezvous::Args& send_args,
                                        const Rendezvous::Args& recv_args,
                                        const Tensor& in, bool is_dead) {
    Status s = status;
    Tensor* out = new Tensor;
    StatusCallback final_callback = [done, send_args, recv_args, out,
                                     is_dead](const Status& s) {
      done(s, send_args, recv_args, *out, is_dead);
      delete out;
    };

    if (s.ok()) {
      SameWorkerRecvDone(parsed, send_args, recv_args, in, out, final_callback);
    } else {
      final_callback(s);
    }
  });
}

void IntraProcessRendezvous::StartAbort(const Status& s) {
  CHECK(!s.ok());
  local_->StartAbort(s);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

}  // end namespace tensorflow
