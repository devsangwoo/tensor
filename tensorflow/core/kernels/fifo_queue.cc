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

// See docs in ../ops/data_flow_ops.cc.

#include <algorithm>
#include <deque>
#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
=======
// See docs in ../ops/data_flow_ops.cc.

#include <deque>
#include <vector>

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fifo_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
<<<<<<< HEAD
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/batch_util.h"
=======
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

FIFOQueue::FIFOQueue(int capacity, const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name)
<<<<<<< HEAD
    : TypedQueue(capacity, component_dtypes, component_shapes, name) {}

void FIFOQueue::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
  DCHECK_GT(queues_[0].size(), size_t{0});
=======
    : QueueBase(component_dtypes, component_shapes, name),
      capacity_(capacity),
      closed_(false) {}

Status FIFOQueue::Initialize() {
  if (component_dtypes_.empty()) {
    return errors::InvalidArgument("Empty component types for queue ", name_);
  }
  if (!component_shapes_.empty() &&
      component_dtypes_.size() != component_shapes_.size()) {
    return errors::InvalidArgument("Different number of component types (",
                                   component_dtypes_.size(), ") vs. shapes (",
                                   component_shapes_.size(), ").");
  }

  mutex_lock lock(mu_);
  queues_.reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    queues_.push_back(SubQueue());
  }
  return Status::OK();
}

// TODO(mrry): If these checks become a bottleneck, find a way to
//   reduce the number of times that they are called.
Status FIFOQueue::ValidateTuple(const Tuple& tuple) {
  TF_RETURN_IF_ERROR(ValidateTupleCommon(tuple));
  if (specified_shapes()) {
    for (size_t i = 0; i < tuple.size(); ++i) {
      if (!tuple[i].shape().IsSameSize(component_shapes_[i])) {
        return errors::InvalidArgument(
            "Shape mismatch in tuple component ", i, ". Expected ",
            component_shapes_[i].ShortDebugString(), ", got ",
            tuple[i].shape().ShortDebugString());
      }
    }
  }
  return Status::OK();
}

// TODO(mrry): If these checks become a bottleneck, find a way to
//   reduce the number of times that they are called.
Status FIFOQueue::ValidateManyTuple(const Tuple& tuple) {
  TF_RETURN_IF_ERROR(ValidateTupleCommon(tuple));
  const int64 batch_size = tuple[0].dim_size(0);
  if (specified_shapes()) {
    for (size_t i = 0; i < tuple.size(); ++i) {
      // Expected shape is [batch_size] + component_shapes_[i]
      const TensorShape expected_shape = ManyOutShape(i, batch_size);
      if (!tuple[i].shape().IsSameSize(expected_shape)) {
        return errors::InvalidArgument(
            "Shape mismatch in tuple component ", i, ". Expected ",
            expected_shape.ShortDebugString(), ", got ",
            tuple[i].shape().ShortDebugString());
      }
    }
  } else {
    for (size_t i = 1; i < tuple.size(); ++i) {
      if (tuple[i].dim_size(0) != batch_size) {
        return errors::InvalidArgument(
            "All input tensors must have the same size in the 0th ",
            "dimension. Component ", i, " has ", tuple[i].dim_size(0),
            ", and should have ", batch_size);
      }
    }
  }
  return Status::OK();
}

void FIFOQueue::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
  DCHECK_GT(queues_[0].size(), 0);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  (*tuple).reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    (*tuple).push_back(*queues_[i][0].AccessTensor(ctx));
    queues_[i].pop_front();
  }
}

<<<<<<< HEAD
=======
void FIFOQueue::Cancel(Action action, CancellationToken token) {
  DoneCallback callback = nullptr;
  {
    mutex_lock lock(mu_);
    std::deque<Attempt>* attempts =
        action == kEnqueue ? &enqueue_attempts_ : &dequeue_attempts_;

    for (Attempt& attempt : *attempts) {
      if (attempt.cancellation_token == token) {
        attempt.is_cancelled = true;
        if (action == kEnqueue) {
          attempt.context->SetStatus(
              errors::Cancelled("Enqueue operation was cancelled"));
        } else {
          attempt.context->SetStatus(
              errors::Cancelled("Dequeue operation was cancelled"));
        }
        std::swap(callback, attempt.done_callback);
        break;
      }
    }
  }
  if (callback) {
    callback();
    FlushUnlocked();
  }
}

void FIFOQueue::CloseAndCancel() {
  std::vector<DoneCallback> callbacks;
  {
    mutex_lock lock(mu_);
    closed_ = true;
    for (Attempt& attempt : enqueue_attempts_) {
      attempt.is_cancelled = true;
      attempt.context->SetStatus(
          errors::Cancelled("Enqueue operation was cancelled"));
      callbacks.emplace_back(std::move(attempt.done_callback));
    }
  }
  for (const DoneCallback& callback : callbacks) {
    callback();
  }
  FlushUnlocked();
}

bool FIFOQueue::TryAttemptLocked(Action action,
                                 std::vector<CleanUp>* clean_up) {
  std::deque<Attempt>* attempts =
      action == kEnqueue ? &enqueue_attempts_ : &dequeue_attempts_;

  bool progress = false;
  bool done = false;
  while (!done && !attempts->empty()) {
    if (attempts->front().is_cancelled) {
      if (action == kEnqueue) {
        LOG(INFO) << "Skipping cancelled enqueue attempt";
      } else {
        LOG(INFO) << "Skipping cancelled dequeue attempt";
      }
      attempts->pop_front();
    } else {
      Attempt* cur_attempt = &attempts->front();
      switch (cur_attempt->run_callback(cur_attempt)) {
        case kNoProgress:
          done = true;
          break;
        case kProgress:
          done = true;
          progress = true;
          break;
        case kComplete:
          progress = true;
          clean_up->emplace_back(std::move(cur_attempt->done_callback),
                                 cur_attempt->cancellation_token,
                                 cur_attempt->context->cancellation_manager());
          attempts->pop_front();
          break;
      }
    }
  }
  return progress;
}

void FIFOQueue::FlushUnlocked() {
  std::vector<CleanUp> clean_up;
  Ref();
  {
    mutex_lock lock(mu_);
    bool changed;
    do {
      changed = TryAttemptLocked(kEnqueue, &clean_up);
      changed = TryAttemptLocked(kDequeue, &clean_up) || changed;
    } while (changed);
  }
  Unref();
  for (const auto& to_clean : clean_up) {
    if (to_clean.to_deregister != CancellationManager::kInvalidToken) {
      // NOTE(mrry): We can safely ignore the return value of
      // DeregisterCallback because the mutex mu_ ensures that the
      // cleanup action only executes once.
      to_clean.cm->DeregisterCallback(to_clean.to_deregister);
    }
    to_clean.finished();
  }
}

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
void FIFOQueue::TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                           DoneCallback callback) {
  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
<<<<<<< HEAD
        token, [this, cm, token]() { Cancel(kEnqueue, cm, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          1, callback, ctx, cm, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("FIFOQueue '", name_, "' is closed."));
=======
        token, [this, token]() { Cancel(kEnqueue, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          1, callback, ctx, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Aborted("FIFOQueue '", name_, "' is closed."));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
              return kComplete;
            }
            if (queues_[0].size() < static_cast<size_t>(capacity_)) {
              for (int i = 0; i < num_components(); ++i) {
                queues_[i].push_back(PersistentTensor(tuple[i]));
              }
              return kComplete;
            } else {
              return kNoProgress;
            }
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Enqueue operation was cancelled"));
    callback();
  }
}

/* static */
Status FIFOQueue::GetElementComponentFromBatch(const FIFOQueue::Tuple& tuple,
<<<<<<< HEAD
                                               int64 index, int component,
=======
                                               int index, int component,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                               OpKernelContext* ctx,
                                               PersistentTensor* out_tensor) {
  TensorShape element_shape(tuple[component].shape());
  element_shape.RemoveDim(0);
  Tensor* element_access = nullptr;
  TF_RETURN_IF_ERROR(ctx->allocate_persistent(
      tuple[component].dtype(), element_shape, out_tensor, &element_access));
  TF_RETURN_IF_ERROR(
<<<<<<< HEAD
      batch_util::CopySliceToElement(tuple[component], element_access, index));
=======
      CopySliceToElement(tuple[component], element_access, index));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return Status::OK();
}

void FIFOQueue::TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                               DoneCallback callback) {
  const int64 batch_size = tuple[0].dim_size(0);
  if (batch_size == 0) {
    callback();
    return;
  }

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
<<<<<<< HEAD
        token, [this, cm, token]() { Cancel(kEnqueue, cm, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          batch_size, callback, ctx, cm, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("FIFOQueue '", name_, "' is closed."));
=======
        token, [this, token]() { Cancel(kEnqueue, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          batch_size, callback, ctx, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Aborted("FIFOQueue '", name_, "' is closed."));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
              return kComplete;
            }
            RunResult result = kNoProgress;
            while (queues_[0].size() < static_cast<size_t>(capacity_)) {
              result = kProgress;
<<<<<<< HEAD
              const int64 index =
=======
              const int index =
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                  tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                PersistentTensor element;
                attempt->context->SetStatus(GetElementComponentFromBatch(
                    tuple, index, i, attempt->context, &element));
                if (!attempt->context->status().ok()) return kComplete;
                queues_[i].push_back(element);
              }
              --attempt->elements_requested;
              if (attempt->elements_requested == 0) {
                return kComplete;
              }
            }
            return result;
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Enqueue operation was cancelled"));
    callback();
  }
}

void FIFOQueue::TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) {
  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
<<<<<<< HEAD
        token, [this, cm, token]() { Cancel(kDequeue, cm, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      dequeue_attempts_.emplace_back(
          1, [callback]() { callback(Tuple()); }, ctx, cm, token,
          [callback, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            const int64 queue_size = queues_[0].size();
            if (closed_ && queue_size == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "FIFOQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ",
                  queue_size, ")"));
              return kComplete;
            }
            if (queue_size > 0) {
=======
        token, [this, token]() { Cancel(kDequeue, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      dequeue_attempts_.emplace_back(
          1, [callback]() { callback(Tuple()); }, ctx, token,
          [callback, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            const int32 s = queues_[0].size();
            if (closed_ && s == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "FIFOQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ", s,
                  ")"));
              return kComplete;
            }
            if (s > 0) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
              Tuple tuple;
              DequeueLocked(attempt->context, &tuple);
              attempt->done_callback = [callback, tuple]() { callback(tuple); };
              return kComplete;
            } else {
              return kNoProgress;
            }
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Dequeue operation was cancelled"));
    callback(Tuple());
  }
}

void FIFOQueue::TryDequeueMany(int num_elements, OpKernelContext* ctx,
<<<<<<< HEAD
                               bool allow_small_batch,
                               CallbackWithTuple callback) {
  if (!specified_shapes()) {
    ctx->SetStatus(errors::InvalidArgument(
        "FIFOQueue's DequeueMany and DequeueUpTo require the "
        "components to have specified shapes."));
=======
                               CallbackWithTuple callback) {
  if (!specified_shapes()) {
    ctx->SetStatus(
        errors::InvalidArgument("FIFOQueue's DequeueMany requires the "
                                "components to have specified shapes."));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    callback(Tuple());
    return;
  }
  if (num_elements == 0) {
    Tuple tuple;
    tuple.reserve(num_components());
    for (int i = 0; i < num_components(); ++i) {
      // TODO(josh11b,misard): Switch to allocate_output().  Problem is
      // this breaks the abstraction boundary since we don't *really*
      // know if and how the Tensors in the tuple we pass to callback
      // correspond to the outputs of *ctx.  For example, the
      // ReaderRead Op uses TryDequeue() to get a filename out of a
      // queue that is used internally by the reader and is not
      // associated with any output of the ReaderRead.
      // mrry@ adds:
      // Maybe we need to pass a std::function<Tensor*(...)> (or
      // better signature) that calls the appropriate allocator
      // function in addition to ctx?  (Or support a shim Allocator
      // that has an internal OpKernelContext*, and dispatches to the
      // appropriate method?)
      // misard@ adds:
      // I don't see that a std::function would help. The problem is
      // that at this point (allocation time) the system doesn't know
      // what is going to happen to the element read out of the
      // queue. As long as we keep the generality that TensorFlow Ops
      // do their own dynamic allocation in arbitrary C++ code, we
      // need to preserve robustness to allocating output Tensors with
      // the 'wrong' attributes, and fixing up with a copy. The only
      // improvement I can see here in the future would be to support
      // an optimized case where the queue 'knows' what attributes to
      // use, and plumbs them through here.
      Tensor element;
<<<<<<< HEAD
      Status status = ctx->allocate_temp(component_dtypes_[i],
                                         ManyOutShape(i, 0), &element);
      if (!status.ok()) {
        ctx->SetStatus(status);
        callback(Tuple());
        return;
      }
=======
      ctx->allocate_temp(component_dtypes_[i], ManyOutShape(i, 0), &element);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      tuple.emplace_back(element);
    }
    callback(tuple);
    return;
  }

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
<<<<<<< HEAD
        token, [this, cm, token]() { Cancel(kDequeue, cm, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      dequeue_attempts_.emplace_back(
          num_elements, [callback]() { callback(Tuple()); }, ctx, cm, token,
          [callback, allow_small_batch,
           this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            int64 queue_size = queues_[0].size();

            if (closed_ && queue_size < attempt->elements_requested) {
              // If we don't have enough for a full dequeue, we have
              // to reset the attempt tuple.
              if (!attempt->tuple.empty()) {
                // Restore already-dequeued elements to the front of the
                // queue.
=======
        token, [this, token]() { Cancel(kDequeue, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      dequeue_attempts_.emplace_back(
          num_elements, [callback]() { callback(Tuple()); }, ctx, token,
          [callback, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            int32 s = queues_[0].size();
            if (closed_ && s < attempt->elements_requested) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "FIFOQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ",
                  attempt->elements_requested, ", current size ", s, ")"));

              // TODO(mrry): Add support for producing a partial batch as
              // output when the queue is closed.
              if (!attempt->tuple.empty()) {
                // Restore already-dequeued elements to the front of the queue.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                for (int64 i = attempt->tuple[0].dim_size(0) -
                               attempt->elements_requested - 1;
                     i >= 0; --i) {
                  for (int j = 0; j < num_components(); ++j) {
                    PersistentTensor element;
                    Status s = GetElementComponentFromBatch(
                        attempt->tuple, i, j, attempt->context, &element);
                    if (!s.ok()) {
                      attempt->context->SetStatus(
                          errors::DataLoss("Failed to restore element from "
                                           "partially-dequeued batch "
<<<<<<< HEAD
                                           "to FIFOQueue: ",
                                           s.error_message()));
=======
                                           "to FIFOQueue"));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                    }
                    queues_[j].push_front(element);
                  }
                }
              }
<<<<<<< HEAD
              if (allow_small_batch && !queues_[0].empty()) {
                // Request all remaining elements in the queue.
                queue_size = queues_[0].size();
                attempt->tuple.clear();
                attempt->elements_requested = queue_size;
              } else {
                if (allow_small_batch) {
                  // There may be some other attempts containing
                  // values.  If so, we'll yield and wait for them
                  // to add elements to the queue.
                  if (!enqueue_attempts_.empty()) return kProgress;
                }
                if (attempt->context->status().ok()) {
                  attempt->context->SetStatus(errors::OutOfRange(
                      "FIFOQueue '", name_, "' is closed and has ",
                      "insufficient elements (requested ",
                      attempt->elements_requested, ", current size ",
                      queue_size, ")"));
                }
                return kComplete;
              }
            }

            RunResult result = kNoProgress;
            for (; queue_size > 0; --queue_size) {
              if (attempt->tuple.empty()) {
                // Only allocate tuple when we have something to dequeue
                // so we don't use excessive memory when there are many
=======
              return kComplete;
            }

            RunResult result = kNoProgress;
            for (; s > 0; --s) {
              if (attempt->tuple.empty()) {
                // Only allocate tuple when we have something to dequeue
                // so we don't use exceessive memory when there are many
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                // blocked dequeue attempts waiting.
                attempt->tuple.reserve(num_components());
                for (int i = 0; i < num_components(); ++i) {
                  const TensorShape shape =
                      ManyOutShape(i, attempt->elements_requested);
                  Tensor element;
<<<<<<< HEAD
                  attempt->context->SetStatus(attempt->context->allocate_temp(
                      component_dtypes_[i], shape, &element));
                  if (!attempt->context->status().ok()) return kComplete;
=======
                  attempt->context->allocate_temp(component_dtypes_[i], shape,
                                                  &element);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                  attempt->tuple.emplace_back(element);
                }
              }
              result = kProgress;
              Tuple tuple;
              DequeueLocked(attempt->context, &tuple);
<<<<<<< HEAD
              const int64 index =
                  attempt->tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                attempt->context->SetStatus(batch_util::CopyElementToSlice(
                    std::move(tuple[i]), &attempt->tuple[i], index));
=======
              const int index =
                  attempt->tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                attempt->context->SetStatus(
                    CopyElementToSlice(tuple[i], &attempt->tuple[i], index));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                if (!attempt->context->status().ok()) return kComplete;
              }
              tuple.clear();
              --attempt->elements_requested;
              if (attempt->elements_requested == 0) {
                tuple = attempt->tuple;
                attempt->done_callback = [callback, tuple]() {
                  callback(tuple);
                };
                return kComplete;
              }
            }
            return result;
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Dequeue operation was cancelled"));
    callback(Tuple());
  }
}

<<<<<<< HEAD
Status FIFOQueue::MatchesNodeDef(const NodeDef& node_def) {
  if (!MatchesNodeDefOp(node_def, "FIFOQueue").ok() &&
      !MatchesNodeDefOp(node_def, "FIFOQueueV2").ok()) {
    return errors::InvalidArgument("Expected FIFOQueue, found ", node_def.op());
  }
=======
void FIFOQueue::Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
                      DoneCallback callback) {
  if (cancel_pending_enqueues) {
    CloseAndCancel();
    callback();
  } else {
    {
      mutex_lock lock(mu_);
      enqueue_attempts_.emplace_back(
          0, callback, ctx, CancellationManager::kInvalidToken,
          [this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(errors::Aborted(
                  "FIFOQueue '", name_, "' is already closed."));
            } else {
              closed_ = true;
            }
            return kComplete;
          });
    }
    FlushUnlocked();
  }
}

Status FIFOQueue::MatchesNodeDef(const NodeDef& node_def) {
  TF_RETURN_IF_ERROR(MatchesNodeDefOp(node_def, "FIFOQueue"));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));
  TF_RETURN_IF_ERROR(MatchesNodeDefTypes(node_def));
  TF_RETURN_IF_ERROR(MatchesNodeDefShapes(node_def));
  return Status::OK();
}

<<<<<<< HEAD
// Defines a FIFOQueueOp, which produces a Queue (specifically, one
// backed by FIFOQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
FIFOQueueOp::FIFOQueueOp(OpKernelConstruction* context)
    : TypedQueueOp(context) {
  OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
}

Status FIFOQueueOp::CreateResource(QueueInterface** ret) {
  FIFOQueue* queue = new FIFOQueue(capacity_, component_types_,
                                   component_shapes_, cinfo_.name());
  return CreateTypedQueue(queue, ret);
}

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
