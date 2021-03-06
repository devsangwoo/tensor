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

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"

#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

namespace {
// The EventMgr has 1 thread for the polling loop and one to execute
// event callback functions. Issues for reconsideration:
//  - Is this the right number of threads?
//  - Should EventMgrs be shared between GPUDevices on a multi-GPU machine?
static const int kNumThreads = 2;
}  // namespace

namespace gpu_event_mgr {
class ThreadLabel {
 public:
  static const char* GetValue() { return value_; }

  // v must be a static const because value_ will capture and use its value
  // until reset or thread terminates.
  static void SetValue(const char* v) { value_ = v; }

 private:
  static thread_local const char* value_;
};
thread_local const char* ThreadLabel::value_ = "";

void WarnIfInCallback(std::function<void()> f) {
  const char* label = ThreadLabel::GetValue();
  if (label && !strcmp(label, "gpu_event_mgr")) {
    if (f) {
      f();
    } else {
      LOG(WARNING) << "Executing inside EventMgr callback thread: "
                   << CurrentStackTrace();
    }
  }
}

void InitThreadpoolLabels(thread::ThreadPool* threadpool) {
  static const char* label = "gpu_event_mgr";
  mutex mu;
  int init_count = 0;
  condition_variable all_initialized;
  int exit_count = 0;
  condition_variable ready_to_exit;
  const int num_threads = threadpool->NumThreads();
  for (int i = 0; i < num_threads; ++i) {
    threadpool->Schedule([num_threads, &mu, &init_count, &all_initialized,
                          &exit_count, &ready_to_exit]() {
      gpu_event_mgr::ThreadLabel::SetValue(label);
      mutex_lock l(mu);
      ++init_count;
      if (init_count == num_threads) {
        all_initialized.notify_all();
      }
      while (init_count < num_threads) {
        all_initialized.wait(l);
      }
      if (++exit_count == num_threads) {
        ready_to_exit.notify_all();
      }
    });
  }
  {
    mutex_lock l(mu);
    while (exit_count < num_threads) {
      ready_to_exit.wait(l);
    }
  }
}
}  // namespace gpu_event_mgr

EventMgr::EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options)
    : exec_(se),
      deferred_bytes_threshold_(gpu_options.deferred_deletion_bytes()
                                    ? gpu_options.deferred_deletion_bytes()
                                    : 8 * 1048576),
      polling_active_delay_usecs_(gpu_options.polling_active_delay_usecs()
                                      ? gpu_options.polling_active_delay_usecs()
                                      : 10),
      accumulated_stream_(nullptr),
      accumulated_tensors_(new TensorReferenceVector),
      accumulated_tensor_bytes_(0),
      threadpool_(Env::Default(), "GPU_Event_Manager", kNumThreads) {
  gpu_event_mgr::InitThreadpoolLabels(&threadpool_);
  StartPollingLoop();
}

EventMgr::~EventMgr() {
  StopPollingLoop();
=======
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/stream.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {

EventMgr::EventMgr(gpu::StreamExecutor* se)
    : exec_(se),
      // threadpool_ has 1 thread for the polling loop, and one to execute
      // event callback functions. Maybe we should have more?
      threadpool_(Env::Default(), "GPU_Event_Manager", 2) {
  threadpool_.Schedule([this]() { PollLoop(); });
}

EventMgr::~EventMgr() {
  stop_polling_.Notify();
  // Shut down the backup polling loop.
  polling_stopped_.WaitForNotification();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  // Events are owned by this object.
  for (auto& e : free_events_) {
    delete e;
  }
<<<<<<< HEAD
  for (auto& t : *(accumulated_tensors_)) {
    t.Unref();
  }
  delete accumulated_tensors_;
  while (!used_events_.empty()) {
    InUse* ue = &used_events_[0];
    delete ue->event;
    if (ue->mem != nullptr) {
      for (auto& t : *(ue->mem)) {
        t.Unref();
      }
      delete ue->mem;
    }
    if (ue->bufrec.buf) {
      if (LogMemory::IsEnabled()) {
        LogMemory::RecordRawDeallocation(ue->bufrec.operation,
                                         ue->bufrec.step_id, ue->bufrec.buf,
                                         ue->bufrec.alloc, false);
      }
      ue->bufrec.alloc->DeallocateRaw(ue->bufrec.buf);
    }
    if (ue->func != nullptr) threadpool_.Schedule(ue->func);
=======
  while (!used_events_.empty()) {
    delete used_events_[0].event;
    delete used_events_[0].mem;
    if (used_events_[0].bufrec.buf) {
      used_events_[0].bufrec.alloc->DeallocateRaw(used_events_[0].bufrec.buf);
    }
    if (used_events_[0].func != nullptr)
      threadpool_.Schedule(used_events_[0].func);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    used_events_.pop_front();
  }
}

<<<<<<< HEAD
void EventMgr::StartPollingLoop() {
  CHECK(polling_stopped_ == nullptr);
  {
    mutex_lock l(mu_);
    stop_polling_ = false;
  }
  polling_stopped_.reset(new Notification);
  threadpool_.Schedule([this]() { PollLoop(); });
}

void EventMgr::StopPollingLoop() {
  if (polling_stopped_) {
    {
      mutex_lock l(mu_);
      stop_polling_ = true;
      events_pending_.notify_all();
    }
    polling_stopped_->WaitForNotification();
    polling_stopped_.reset(nullptr);
  }
}

void EventMgr::ThenDeleteTensors(se::Stream* stream,
                                 const TensorReferenceVector& tensors) {
  mutex_lock l(mu_);
  // TODO(jeff): We currently keep one accumulated_tensors_ object.
  // If we start to use multiple streams heavily, we might want to keep
  // separate vectors/byte counters per stream
  if (!accumulated_tensors_->empty() && stream != accumulated_stream_) {
    FlushAccumulatedTensors();
  }
  accumulated_stream_ = stream;
  for (const auto& t : tensors) {
    // accumulated_tensors_ takes over ownership of the reference to "t"
    accumulated_tensors_->push_back(t);
    accumulated_tensor_bytes_ += t.TotalBytes();
  }
  if (accumulated_tensor_bytes_ >= deferred_bytes_threshold_) {
    FlushAccumulatedTensors();
  }
}

void EventMgr::FlushAccumulatedTensors() {
  DCHECK(!accumulated_tensors_->empty());
  DCHECK(accumulated_stream_ != nullptr);
  QueueTensors(accumulated_stream_, accumulated_tensors_);
  accumulated_tensors_ = new TensorReferenceVector;
  accumulated_tensor_bytes_ = 0;
  accumulated_stream_ = nullptr;
}

// A polling loop to detect completion of GPU events.
//
// While one or more events is outstanding, poll for completed events.  When no
// events are outstanding, we sleep until one is enqueued.
void EventMgr::PollLoop() {
  ToFreeVector to_free;
  while (true) {
    bool events_still_pending;
    {
      mutex_lock l(mu_);
      if (stop_polling_) {
        break;
      }
      if (used_events_.empty()) {
        events_pending_.wait(l);
      }
      PollEvents(true, &to_free);
      events_still_pending = !used_events_.empty();
    }
    FreeMemory(to_free);
    to_free.clear();

    if (events_still_pending) {
      Env::Default()->SleepForMicroseconds(polling_active_delay_usecs_);
    }
  }
  polling_stopped_->Notify();
}

void EventMgr::QueueInUse(se::Stream* stream, InUse iu) {
=======
// This polling loop runs at a relatively low frequency. Most calls to
// PollEvents() should come directly from Compute() via
// ThenDeleteTensors().  This function's purpose is to ensure that
// even if no more GPU operations are being requested, we still
// eventually clear the queue. It seems to prevent some tensorflow
// programs from stalling for reasons not yet understood.
void EventMgr::PollLoop() {
  while (!stop_polling_.HasBeenNotified()) {
    Env::Default()->SleepForMicroseconds(1 * 1000);
    {
      mutex_lock l(mu_);
      PollEvents(true);
    }
  }
  polling_stopped_.Notify();
}

void EventMgr::QueueInUse(gpu::Stream* stream, InUse iu) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  VLOG(2) << "QueueInUse  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Events are created on demand, and repeatedly reused.  There is no
  // limit placed here on the number of allocated Events.
  if (free_events_.empty()) {
<<<<<<< HEAD
    free_events_.push_back(new se::Event(exec_));
    free_events_.back()->Init();
  }
  se::Event* e = free_events_.back();
  free_events_.pop_back();
  stream->ThenRecordEvent(e);
  iu.event = e;
  bool was_empty = used_events_.empty();
  used_events_.push_back(iu);
  // Maybe wake up the polling thread
  if (was_empty) events_pending_.notify_all();
=======
    free_events_.push_back(new gpu::Event(exec_));
    free_events_.back()->Init();
  }
  gpu::Event* e = free_events_.back();
  free_events_.pop_back();
  stream->ThenRecordEvent(e);
  iu.event = e;
  used_events_.push_back(iu);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

// This function must be called periodically to check whether pending
// events have recorded, and then retire them.  Initial observations
// suggest that typical behavior in a TensorFlow program is to have
// 0-3 events pending most of the time, but there are occasionally
<<<<<<< HEAD
// spikes of up to several hundred outstanding.  (If GPUKernelTracker
// is used to cap pending kernels there should never be more than
// that many.)
=======
// spikes of up to several hundred outstanding.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
//
// NOTE: If all events are on the same stream, no later event will
// complete before an earlier event, except possibly if the earlier
// event transitions to an error state, so there's no advantage in
// looking past the first kPending event.  However, if we're using
// multiple streams there may be some gain in looking deeper.
// As a compromise, PollEvent() calls that are triggered by the queueing
<<<<<<< HEAD
// of a single event never look past the first kPending event.  Consequently
// those calls do an expected constant amount of work, unaffected by the
// length of the pending queue.  Calls coming from the dedicated
// polling thread always sweep the full queue.
void EventMgr::PollEvents(bool is_dedicated_poller,
                          gtl::InlinedVector<InUse, 4>* to_free) {
=======
// of a single event never look past the first kPending event.  Calls
// coming from the dedicated polling thread always sweep the full queue.
//
// Note that allowing the queue to grow very long could cause overall
// GPU memory use to spike needlessly.  An alternative strategy would
// be to throttle new Op execution until the pending event queue
// clears.
void EventMgr::PollEvents(bool is_dedicated_poller) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  VLOG(2) << "PollEvents  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Sweep the remaining events in order.  If this is the dedicated
  // polling thread, check the entire set.  Otherwise, just sweep up to
  // the first non-complete record that is still pending.
  for (auto& iu : used_events_) {
    if (iu.event == nullptr) continue;
<<<<<<< HEAD
    se::Event::Status s = iu.event->PollForStatus();
    switch (s) {
      case se::Event::Status::kUnknown:
      case se::Event::Status::kError:
=======
    gpu::Event::Status s = iu.event->PollForStatus();
    switch (s) {
      case gpu::Event::Status::kUnknown:
      case gpu::Event::Status::kError:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
        // We don't expect to see these.  Someday maybe propagate
        // a Status error, but for now fail hard.
        LOG(FATAL) << "Unexpected Event status: " << static_cast<int>(s);
        break;
<<<<<<< HEAD
      case se::Event::Status::kPending:
        if (!is_dedicated_poller) return;  // quit processing queue
        break;
      case se::Event::Status::kComplete:
        // Make a copy of the InUse record so we can free it after releasing
        // the lock
        to_free->push_back(iu);
=======
      case gpu::Event::Status::kPending:
        if (!is_dedicated_poller) return;  // quit processing queue
        break;
      case gpu::Event::Status::kComplete:
        delete iu.mem;
        if (iu.bufrec.buf) iu.bufrec.alloc->DeallocateRaw(iu.bufrec.buf);
        // The function must be called in another thread, outside of
        // the mutex held here.
        if (iu.func != nullptr) threadpool_.Schedule(iu.func);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
        free_events_.push_back(iu.event);
        // Mark this InUse record as completed.
        iu.event = nullptr;
    }
  }
  // Then clear any completed InUse records from the front of the queue.
  while (!used_events_.empty()) {
    InUse& iu = used_events_.front();
    if (iu.event == nullptr) {
      used_events_.pop_front();
    } else {
      break;
    }
  }
}

<<<<<<< HEAD
EventMgrFactory* EventMgrFactory::Singleton() {
  static EventMgrFactory* instance = new EventMgrFactory;
  return instance;
}

EventMgr* EventMgrFactory::GetEventMgr(se::StreamExecutor* se,
                                       const GPUOptions& gpu_options) {
  mutex_lock l(mu_);
  // TODO(laigd): consider making gpu_options part of the key. It's not
  // currently since EventMgr depends only rely on field deferred_deletion_bytes
  // and polling_active_delay_usecs from gpu_options which are not used or
  // rarely used.
  auto itr = event_mgr_map_.find(se);
  if (itr == event_mgr_map_.end()) {
    auto event_mgr = new EventMgr(se, gpu_options);
    event_mgr_map_[se] = event_mgr;
    return event_mgr;
  } else {
    return itr->second;
  }
}

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
