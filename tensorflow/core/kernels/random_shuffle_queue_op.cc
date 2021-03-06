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
// See docs in ../ops/data_flow_ops.cc.

#include <deque>
#include <vector>

<<<<<<< HEAD
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/kernels/typed_queue.h"
=======
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_base.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
<<<<<<< HEAD
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {

class RandomShuffleQueue : public TypedQueue<std::vector<PersistentTensor> > {
=======
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

class RandomShuffleQueue : public QueueBase {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
 public:
  RandomShuffleQueue(int32 capacity, int32 min_after_dequeue, int64 seed,
                     int64 seed2, const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name);
<<<<<<< HEAD

  Status Initialize() override;  // Must be called before any other method.

  // Implementations of QueueInterface methods --------------------------------
=======
  Status Initialize();  // Must be called before any other method.

  // Implementations of QueueInterface methods --------------------------------

  Status ValidateTuple(const Tuple& tuple) override;
  Status ValidateManyTuple(const Tuple& tuple) override;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
<<<<<<< HEAD
                      bool allow_small_batch,
                      CallbackWithTuple callback) override;
  Status MatchesNodeDef(const NodeDef& node_def) override;

  int32 size() const override {
=======
                      CallbackWithTuple callback) override;
  void Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
             DoneCallback callback) override;
  Status MatchesNodeDef(const NodeDef& node_def) override;

  int32 size() override {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    mutex_lock lock(mu_);
    return queues_[0].size();
  }

 private:
<<<<<<< HEAD
  ~RandomShuffleQueue() override {}

=======
  enum Action { kEnqueue, kDequeue };

  ~RandomShuffleQueue() override {}

  TensorShape ManyOutShape(int i, int batch_size) {
    TensorShape shape({batch_size});
    shape.AppendShape(component_shapes_[i]);
    return shape;
  }

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // Helper for dequeuing a single random element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

<<<<<<< HEAD
  static Status GetElementComponentFromBatch(const Tuple& tuple, int64 index,
                                             int component,
                                             OpKernelContext* ctx,
                                             PersistentTensor* out_tensor);

=======
  void Cancel(Action action, CancellationToken token);

  // Helper for cancelling all pending Enqueue(Many) operations when
  // Close is called with cancel_pending_enqueues.
  void CloseAndCancel();

  // Tries to enqueue/dequeue (or close) based on whatever is at the
  // front of enqueue_attempts_/dequeue_attempts_.  Appends to
  // *finished the callback for any finished attempt (so it may be
  // called once mu_ is released).  Returns true if any progress was
  // made.
  struct CleanUp {
    CleanUp(DoneCallback&& f, CancellationToken ct, CancellationManager* cm)
        : finished(f), to_deregister(ct), cm(cm) {}
    DoneCallback finished;
    CancellationToken to_deregister;
    CancellationManager* cm;
  };
  bool TryAttemptLocked(Action action, std::vector<CleanUp>* clean_up)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Tries to make progress on the enqueues or dequeues at the front
  // of the *_attempts_ queues.
  void FlushUnlocked();

  const int32 capacity_;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  const int32 min_after_dequeue_;
  const int64 original_seed_;
  const int64 original_seed2_;

<<<<<<< HEAD
  random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
  random::SingleSampleAdapter<random::PhiloxRandom> generator_ GUARDED_BY(mu_);

=======
  mutex mu_;
  typedef std::vector<PersistentTensor> SubQueue;
  std::vector<SubQueue> queues_ GUARDED_BY(mu_);
  bool closed_ GUARDED_BY(mu_);
  random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
  random::SingleSampleAdapter<random::PhiloxRandom> generator_ GUARDED_BY(mu_);

  enum RunResult { kNoProgress, kProgress, kComplete };
  struct Attempt;
  typedef std::function<RunResult(Attempt*)> RunCallback;
  struct Attempt {
    int32 elements_requested;
    DoneCallback done_callback;  // must be run outside mu_
    OpKernelContext* context;
    CancellationToken cancellation_token;
    RunCallback run_callback;  // must be run while holding mu_
    bool is_cancelled;
    Tuple tuple;

    Attempt(int32 elements_requested, DoneCallback done_callback,
            OpKernelContext* context, CancellationToken cancellation_token,
            RunCallback run_callback)
        : elements_requested(elements_requested),
          done_callback(done_callback),
          context(context),
          cancellation_token(cancellation_token),
          run_callback(run_callback),
          is_cancelled(false) {}
  };
  std::deque<Attempt> enqueue_attempts_ GUARDED_BY(mu_);
  std::deque<Attempt> dequeue_attempts_ GUARDED_BY(mu_);

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  TF_DISALLOW_COPY_AND_ASSIGN(RandomShuffleQueue);
};

RandomShuffleQueue::RandomShuffleQueue(
<<<<<<< HEAD
    int32 capacity, int32 min_after_dequeue, int64 seed, int64 seed2,
    const DataTypeVector& component_dtypes,
    const std::vector<TensorShape>& component_shapes, const string& name)
    : TypedQueue(capacity, component_dtypes, component_shapes, name),
      min_after_dequeue_(min_after_dequeue),
      original_seed_(seed),
      original_seed2_(seed2),
=======
    int capacity, int min_after_dequeue, int64 seed, int64 seed2,
    const DataTypeVector& component_dtypes,
    const std::vector<TensorShape>& component_shapes, const string& name)
    : QueueBase(component_dtypes, component_shapes, name),
      capacity_(capacity),
      min_after_dequeue_(min_after_dequeue),
      original_seed_(seed),
      original_seed2_(seed2),
      closed_(false),
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      generator_(&parent_generator_) {
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, use completely random seeds.
    seed = random::New64();
    seed2 = random::New64();
  }
  parent_generator_ = random::PhiloxRandom(seed, seed2);
}

Status RandomShuffleQueue::Initialize() {
<<<<<<< HEAD
  TF_RETURN_IF_ERROR(TypedQueue::Initialize());

  mutex_lock lock(mu_);
  for (int i = 0; i < num_components(); ++i) {
    queues_[i].reserve(min_after_dequeue_);
=======
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
    queues_.back().reserve(min_after_dequeue_);
  }
  return Status::OK();
}

// TODO(mrry): If these checks become a bottleneck, find a way to
//   reduce the number of times that they are called.
Status RandomShuffleQueue::ValidateTuple(const Tuple& tuple) {
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
Status RandomShuffleQueue::ValidateManyTuple(const Tuple& tuple) {
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
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
  return Status::OK();
}

void RandomShuffleQueue::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
<<<<<<< HEAD
  DCHECK_GT(queues_[0].size(), size_t{0});
=======
  DCHECK_GT(queues_[0].size(), 0);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  int64 index = generator_() % queues_[0].size();
  (*tuple).reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    (*tuple).push_back(*queues_[i][index].AccessTensor(ctx));
    queues_[i][index] = queues_[i].back();
    queues_[i].pop_back();
  }
}

<<<<<<< HEAD
=======
void RandomShuffleQueue::Cancel(Action action, CancellationToken token) {
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

void RandomShuffleQueue::CloseAndCancel() {
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

bool RandomShuffleQueue::TryAttemptLocked(
    Action action, std::vector<CleanUp>* clean_up) {
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

void RandomShuffleQueue::FlushUnlocked() {
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
void RandomShuffleQueue::TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
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
              attempt->context->SetStatus(errors::Cancelled(
=======
        token, [this, token]() { Cancel(kEnqueue, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          1, callback, ctx, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(errors::Aborted(
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                  "RandomShuffleQueue '", name_, "' is closed."));
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

<<<<<<< HEAD
/* static */
Status RandomShuffleQueue::GetElementComponentFromBatch(
    const Tuple& tuple, int64 index, int component, OpKernelContext* ctx,
    PersistentTensor* out_tensor) {
  TensorShape element_shape(tuple[component].shape());
  element_shape.RemoveDim(0);
  Tensor* element_access = nullptr;
  TF_RETURN_IF_ERROR(ctx->allocate_persistent(
      tuple[component].dtype(), element_shape, out_tensor, &element_access));
  TF_RETURN_IF_ERROR(
      batch_util::CopySliceToElement(tuple[component], element_access, index));
  return Status::OK();
}

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
void RandomShuffleQueue::TryEnqueueMany(const Tuple& tuple,
                                        OpKernelContext* ctx,
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
              attempt->context->SetStatus(errors::Cancelled(
=======
        token, [this, token]() { Cancel(kEnqueue, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          batch_size, callback, ctx, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(errors::Aborted(
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                  "RandomShuffleQueue '", name_, "' is closed."));
              return kComplete;
            }
            RunResult result = kNoProgress;
            while (queues_[0].size() < static_cast<size_t>(capacity_)) {
              result = kProgress;
              const int index =
                  tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
<<<<<<< HEAD
                PersistentTensor element;
                attempt->context->SetStatus(GetElementComponentFromBatch(
                    tuple, index, i, attempt->context, &element));
=======
                TensorShape element_shape(tuple[i].shape());
                element_shape.RemoveDim(0);
                PersistentTensor element;
                Tensor* element_access = nullptr;
                attempt->context->allocate_persistent(
                    tuple[i].dtype(), element_shape, &element, &element_access);
                attempt->context->SetStatus(
                    CopySliceToElement(tuple[i], element_access, index));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
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

void RandomShuffleQueue::TryDequeue(OpKernelContext* ctx,
                                    CallbackWithTuple callback) {
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
            int32 queue_size = queues_[0].size();
            if (closed_ && queue_size == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "RandomShuffleQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ",
                  queue_size, ")"));
              return kComplete;
            }
            if (!closed_) queue_size -= min_after_dequeue_;
            if (queue_size > 0) {
=======
        token, [this, token]() { Cancel(kDequeue, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      dequeue_attempts_.emplace_back(
          1, [callback]() { callback(Tuple()); }, ctx, token,
          [callback, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            int32 s = queues_[0].size();
            if (closed_ && s == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "RandomShuffleQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ", s,
                  ")"));
              return kComplete;
            }
            if (!closed_) s -= min_after_dequeue_;
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

void RandomShuffleQueue::TryDequeueMany(int num_elements, OpKernelContext* ctx,
<<<<<<< HEAD
                                        bool allow_small_batch,
                                        CallbackWithTuple callback) {
  if (!specified_shapes()) {
    ctx->SetStatus(errors::InvalidArgument(
        "RandomShuffleQueue's DequeueMany and DequeueUpTo require the "
        "components to have specified shapes."));
=======
                                        CallbackWithTuple callback) {
  if (!specified_shapes()) {
    ctx->SetStatus(
        errors::InvalidArgument("RandomShuffleQueue's DequeueMany requires the "
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
      Status s = ctx->allocate_temp(component_dtypes_[i], ManyOutShape(i, 0),
                                    &element);
      if (!s.ok()) {
        ctx->SetStatus(s);
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
            int32 queue_size = queues_[0].size();
            if (closed_ && queue_size < attempt->elements_requested) {
              // If we don't have enough for a full dequeue, we have
              // to reset the attempt tuple.
              if (!attempt->tuple.empty()) {
                // Restore already-dequeued elements to the queue.
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
                                           "to RandomShuffleQueue: ",
                                           s.error_message()));
                    }
                    queues_[j].push_back(element);
                  }
                }
              }
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
                      "RandomShuffleQueue '", name_, "' is closed and has ",
                      "insufficient elements (requested ",
                      attempt->elements_requested, ", current size ",
                      queue_size, ")"));
                }
                return kComplete;
              }
            }

            RunResult result = kNoProgress;
            if (!closed_) queue_size -= min_after_dequeue_;
            for (; queue_size > 0; --queue_size) {
              if (attempt->tuple.empty()) {
                // Only allocate tuple when we have something to dequeue
                // so we don't use excessive memory when there are many
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
                  "RandomSuffleQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ",
                  attempt->elements_requested, ", current size ", s, ")"));
              return kComplete;
            }

            RunResult result = kNoProgress;
            if (!closed_) s -= min_after_dequeue_;
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
              const int index =
                  attempt->tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
<<<<<<< HEAD
                attempt->context->SetStatus(batch_util::CopyElementToSlice(
                    std::move(tuple[i]), &attempt->tuple[i], index));
=======
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
Status RandomShuffleQueue::MatchesNodeDef(const NodeDef& node_def) {
  if (!MatchesNodeDefOp(node_def, "RandomShuffleQueue").ok() &&
      !MatchesNodeDefOp(node_def, "RandomShuffleQueueV2").ok()) {
    return errors::InvalidArgument("Expected RandomShuffleQueue, found ",
                                   node_def.op());
  }
=======
void RandomShuffleQueue::Close(OpKernelContext* ctx,
                               bool cancel_pending_enqueues,
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
                  "RandomShuffleQueue '", name_, "' is already closed."));
            } else {
              closed_ = true;
            }
            return kComplete;
          });
    }
    FlushUnlocked();
  }
}

Status RandomShuffleQueue::MatchesNodeDef(const NodeDef& node_def) {
  TF_RETURN_IF_ERROR(MatchesNodeDefOp(node_def, "RandomShuffleQueue"));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));

  int32 min_after_dequeue = -1;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node_def, "min_after_dequeue", &min_after_dequeue));
  if (min_after_dequeue != min_after_dequeue_) {
    return errors::InvalidArgument(
<<<<<<< HEAD
        "Shared queue '", name_, "' has min_after_dequeue ", min_after_dequeue_,
        " but requested min_after_dequeue was ", min_after_dequeue, ".");
=======
        "Shared queue '", name_, "' has min_after_dequeue ",
        min_after_dequeue_, " but requested min_after_dequeue was ",
        min_after_dequeue, ".");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  int64 seed = -1;
  int64 seed2 = -1;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "seed", &seed));
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "seed2", &seed2));
  if ((seed != 0 || seed2 != 0) &&
      (seed != original_seed_ || seed2 != original_seed2_)) {
    return errors::InvalidArgument(
        "Shared queue '", name_, "' has random seeds (", original_seed_, ", ",
        original_seed2_, ") but requested seeds are (", seed, ", ", seed2,
        ").");
  }

  TF_RETURN_IF_ERROR(MatchesNodeDefTypes(node_def));
  TF_RETURN_IF_ERROR(MatchesNodeDefShapes(node_def));

  return Status::OK();
}

<<<<<<< HEAD
=======
typedef std::shared_ptr<QueueInterface> QueueInterfacePtr;

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// Defines a RandomShuffleQueueOp, which produces a Queue (specifically, one
// backed by RandomShuffleQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
<<<<<<< HEAD
class RandomShuffleQueueOp : public TypedQueueOp {
 public:
  explicit RandomShuffleQueueOp(OpKernelConstruction* context)
      : TypedQueueOp(context) {
=======
class RandomShuffleQueueOp : public OpKernel {
 public:
  explicit RandomShuffleQueueOp(OpKernelConstruction* context)
      : OpKernel(context), queue_handle_set_(false) {
    OP_REQUIRES_OK(context, context->GetAttr("capacity", &capacity_));
    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                &queue_handle_, nullptr));
    if (capacity_ < 0) {
      capacity_ = RandomShuffleQueue::kUnbounded;
    }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    OP_REQUIRES_OK(context,
                   context->GetAttr("min_after_dequeue", &min_after_dequeue_));
    OP_REQUIRES(context, min_after_dequeue_ >= 0,
                errors::InvalidArgument("min_after_dequeue ",
                                        min_after_dequeue_, " must be >= 0"));
    OP_REQUIRES(
        context, min_after_dequeue_ < capacity_,
        errors::InvalidArgument("min_after_dequeue ", min_after_dequeue_,
                                " must be < capacity ", capacity_));
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(context, context->GetAttr("seed2", &seed2_));

<<<<<<< HEAD
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
  }

 private:
  Status CreateResource(QueueInterface** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    RandomShuffleQueue* queue = new RandomShuffleQueue(
        capacity_, min_after_dequeue_, seed_, seed2_, component_types_,
        component_shapes_, cinfo_.name());
    return CreateTypedQueue(queue, ret);
  }

  int32 min_after_dequeue_;
  int64 seed_;
  int64 seed2_;
  std::vector<TensorShape> component_shapes_;
=======
    OP_REQUIRES_OK(context,
                   context->GetAttr("component_types", &component_types_));
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
  }

  ~RandomShuffleQueueOp() override {
    // If the queue object was not shared, delete it.
    if (queue_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK(cinfo_.resource_manager()->Delete<QueueInterface>(
          cinfo_.container(), cinfo_.name()));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    if (!queue_handle_set_) {
      OP_REQUIRES_OK(ctx, SetQueueHandle(ctx));
    }
    ctx->set_output_ref(0, &mu_, queue_handle_.AccessTensor(ctx));
  }

 private:
  Status SetQueueHandle(OpKernelContext* ctx) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
    QueueInterface* queue;
    auto creator = [this](QueueInterface** ret) {
      auto* q = new RandomShuffleQueue(capacity_, min_after_dequeue_, seed_,
                                       seed2_, component_types_,
                                       component_shapes_, cinfo_.name());
      Status s = q->Initialize();
      if (s.ok()) {
        *ret = q;
      } else {
        q->Unref();
      }
      return s;
    };
    TF_RETURN_IF_ERROR(
        cinfo_.resource_manager()->LookupOrCreate<QueueInterface>(
            cinfo_.container(), cinfo_.name(), &queue, creator));
    core::ScopedUnref unref_me(queue);
    // Verify that the shared queue is compatible with the requested arguments.
    TF_RETURN_IF_ERROR(queue->MatchesNodeDef(def()));
    auto h = queue_handle_.AccessTensor(ctx)->flat<string>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    queue_handle_set_ = true;
    return Status::OK();
  }

  int32 capacity_;
  int32 min_after_dequeue_;
  int64 seed_;
  int64 seed2_;
  DataTypeVector component_types_;
  std::vector<TensorShape> component_shapes_;
  ContainerInfo cinfo_;

  mutex mu_;
  PersistentTensor queue_handle_ GUARDED_BY(mu_);
  bool queue_handle_set_ GUARDED_BY(mu_);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  TF_DISALLOW_COPY_AND_ASSIGN(RandomShuffleQueueOp);
};

REGISTER_KERNEL_BUILDER(Name("RandomShuffleQueue").Device(DEVICE_CPU),
                        RandomShuffleQueueOp);
<<<<<<< HEAD
REGISTER_KERNEL_BUILDER(Name("RandomShuffleQueueV2").Device(DEVICE_CPU),
                        RandomShuffleQueueOp);
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace tensorflow
