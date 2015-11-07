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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_

#include <deque>
#include <vector>
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace stream_executor {
class Event;
class Stream;
class StreamExecutor;
}  // namespace stream_executor

namespace tensorflow {

class GPUOptions;

// The callback provided to EventMgr::ThenExecute must not block or take a long
// time.  If it does, performance may be impacted and GPU memory may be
// exhausted.  This macro is for checking that an EventMgr thread is not
// accidentally entering blocking parts of the code, e.g. the RPC subsystem.
//
// Intended use is something like
//
//   void RespondToAnRPC(Params* params) {
//      WARN_IF_IN_EVENT_MGR_THREAD;
//      if (params->status.ok()) { ...
//
namespace gpu_event_mgr {
// Logs a stack trace if current execution thread belongs to this EventMgr
// object.  If f is not nullptr, executes instead of  logging the stack trace.
// trace.
void WarnIfInCallback(std::function<void()> f);
}  // namespace gpu_event_mgr
#define WARN_IF_IN_EVENT_MGR_THREAD gpu_event_mgr::WarnIfInCallback(nullptr)

=======
#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_

#include <deque>
#include <vector>
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/tensor.h"

namespace perftools {
namespace gputools {
class Event;
class Stream;
class StreamExecutor;
}  // namespace gputools
}  // namespace perftools

namespace tensorflow {

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// An object to keep track of pending Events in the StreamExecutor streams
// and associated Tensors that cannot safely be deleted until the associated
// Events are recorded.
class EventMgr {
 public:
<<<<<<< HEAD
  virtual ~EventMgr();

  // Releases the references on the elements of "tensors" as soon as
  // all events currently enqueued on "stream" have completed.
  void ThenDeleteTensors(se::Stream* stream,
                         const TensorReferenceVector& tensors);
=======
  explicit EventMgr(perftools::gputools::StreamExecutor* se);

  ~EventMgr();

  // Takes ownership of *tensors and deletes it as soon as all events
  // currently enqueued on *stream have completed.
  inline void ThenDeleteTensors(perftools::gputools::Stream* stream,
                                std::vector<Tensor>* tensors) {
    mutex_lock l(mu_);
    QueueTensors(stream, tensors);
    PollEvents(false);
  }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  struct BufRec {
    Allocator* alloc;
    void* buf;
<<<<<<< HEAD
    // operation and step_id are only populated when
    // LogMemory::IsEnabled() is true.
    string operation;
    int64 step_id;
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  };

  // Takes ownership of *bufrec.buf and calls bufrec.alloc->DeallocateRaw()
  // on it as soon as all events currently enqueued on *stream have completed.
<<<<<<< HEAD
  inline void ThenDeleteBuffer(se::Stream* stream, BufRec bufrec) {
    ToFreeVector to_free;
    {
      mutex_lock l(mu_);
      QueueBuffer(stream, bufrec);
      PollEvents(false, &to_free);
    }
    FreeMemory(to_free);
  }

  // Execute func when all pending stream actions have completed.
  // func must be brief and non-blocking since it executes in the one
  // thread used for all such callbacks and also buffer deletions.
  inline void ThenExecute(se::Stream* stream, std::function<void()> func) {
    ToFreeVector to_free;
    {
      mutex_lock l(mu_);
      QueueFunc(stream, std::move(func));
      PollEvents(false, &to_free);
    }
    FreeMemory(to_free);
  }

 private:
  friend class TEST_EventMgr;
  friend class TEST_EventMgrHelper;
  friend class EventMgrFactory;
  se::StreamExecutor* const exec_;
  const int64 deferred_bytes_threshold_;
  const int32 polling_active_delay_usecs_;
  mutex mu_;
  condition_variable events_pending_ GUARDED_BY(mu_);

  void FlushAccumulatedTensors() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  struct InUse {
    se::Event* event;
    TensorReferenceVector* mem;
=======
  inline void ThenDeleteBuffer(perftools::gputools::Stream* stream,
                               BufRec bufrec) {
    mutex_lock l(mu_);
    QueueBuffer(stream, bufrec);
    PollEvents(false);
  }

  inline void ThenExecute(perftools::gputools::Stream* stream,
                          std::function<void()> func) {
    mutex_lock l(mu_);
    QueueFunc(stream, func);
    PollEvents(false);
  }

 private:
  friend class TEST_EventMgrHelper;
  mutex mu_;
  perftools::gputools::StreamExecutor* exec_;

  struct InUse {
    perftools::gputools::Event* event;
    std::vector<Tensor>* mem;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    BufRec bufrec;
    std::function<void()> func;
  };

<<<<<<< HEAD
  typedef gtl::InlinedVector<InUse, 4> ToFreeVector;

  EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);

  void FreeMemory(const ToFreeVector& to_free) {
    for (const auto& iu : to_free) {
      if (iu.mem != nullptr) {
        for (auto& t : *(iu.mem)) {
          t.Unref();
        }
        delete iu.mem;
      }
      if (iu.bufrec.buf) {
        if (LogMemory::IsEnabled()) {
          LogMemory::RecordRawDeallocation(iu.bufrec.operation,
                                           iu.bufrec.step_id, iu.bufrec.buf,
                                           iu.bufrec.alloc, false);
        }
        iu.bufrec.alloc->DeallocateRaw(iu.bufrec.buf);
      }
      // The function must be called in another thread.
      if (iu.func != nullptr) threadpool_.Schedule(iu.func);
    }
  }

  // Stream-enqueue an unused Event and save with it a collection of
  // Tensors and/or a BufRec to be deleted only after the Event
  // records.
  void QueueInUse(se::Stream* stream, InUse in_use)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void QueueTensors(se::Stream* stream, TensorReferenceVector* tensors)
=======
  // Stream-enqueue an unused Event and save with it a collection of
  // Tensors and/or a BufRec to be deleted only after the Event
  // records.
  void QueueInUse(perftools::gputools::Stream* stream, InUse in_use)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void QueueTensors(perftools::gputools::Stream* stream,
                    std::vector<Tensor>* tensors)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, tensors, BufRec(), nullptr});
  }

<<<<<<< HEAD
  void QueueBuffer(se::Stream* stream, BufRec bufrec)
=======
  void QueueBuffer(perftools::gputools::Stream* stream, BufRec bufrec)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, nullptr, bufrec, nullptr});
  }

<<<<<<< HEAD
  void QueueFunc(se::Stream* stream, std::function<void()> func)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, nullptr, BufRec(), std::move(func)});
=======
  void QueueFunc(perftools::gputools::Stream* stream,
                 std::function<void()> func) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, nullptr, BufRec(), func});
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  // This function should be called at roughly the same tempo as
  // QueueTensors() to check whether pending events have recorded,
<<<<<<< HEAD
  // and then retire them.  It appends InUse elements that need cleanup
  // to "*to_free".  The caller should call FreeMemory(to_free)
  // when this returns.
  void PollEvents(bool is_dedicated_poller, ToFreeVector* to_free)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);
=======
  // and then retire them.
  void PollEvents(bool is_dedicated_poller) EXCLUSIVE_LOCKS_REQUIRED(mu_);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // An internal polling loop that runs at a low frequency to clear
  // straggler Events.
  void PollLoop();

<<<<<<< HEAD
  // Setup/Teardown functions for the polling loop.
  void StartPollingLoop();
  void StopPollingLoop();

  // A stack of unused events
  std::vector<se::Event*> free_events_ GUARDED_BY(mu_);

  // Buffered list of tensors waiting to have an event queued for deletion
  se::Stream* accumulated_stream_ GUARDED_BY(mu_);
  TensorReferenceVector* accumulated_tensors_ GUARDED_BY(mu_);
  // Sum of the TotalBytes() of the tensors in "accumulated_tensors_"
  int64 accumulated_tensor_bytes_ GUARDED_BY(mu_);
=======
  // A stack of unused events
  std::vector<perftools::gputools::Event*> free_events_ GUARDED_BY(mu_);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // A FIFO queue of InUse events and associated tensors.
  std::deque<InUse> used_events_ GUARDED_BY(mu_);

<<<<<<< HEAD
  bool stop_polling_ GUARDED_BY(mu_);
  std::unique_ptr<Notification> polling_stopped_;
=======
  Notification stop_polling_;
  Notification polling_stopped_;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // The main PollLoop for the event manager runs in this threadpool.
  thread::ThreadPool threadpool_;
};

<<<<<<< HEAD
// Manages all the EventMgr instances.
class EventMgrFactory {
 public:
  static EventMgrFactory* Singleton();

  EventMgr* GetEventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);

 private:
  mutex mu_;

  // Maintain one EventMgr per physical device (StreamExecutor is
  // per-physical-device).
  std::map<se::StreamExecutor*, EventMgr*> event_mgr_map_ GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
=======
}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
