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

#ifndef TENSORFLOW_CORE_KERNELS_FIFO_QUEUE_H_
#define TENSORFLOW_CORE_KERNELS_FIFO_QUEUE_H_
=======
#ifndef TENSORFLOW_KERNELS_FIFO_QUEUE_H_
#define TENSORFLOW_KERNELS_FIFO_QUEUE_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/kernels/typed_queue.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class FIFOQueue : public TypedQueue<std::deque<PersistentTensor> > {
=======
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

class FIFOQueue : public QueueBase {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
 public:
  FIFOQueue(int32 capacity, const DataTypeVector& component_dtypes,
            const std::vector<TensorShape>& component_shapes,
            const string& name);
<<<<<<< HEAD

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

<<<<<<< HEAD
 protected:
  ~FIFOQueue() override {}

=======
  int32 capacity() const { return capacity_; }

 private:
  enum Action { kEnqueue, kDequeue };

  ~FIFOQueue() override {}

  TensorShape ManyOutShape(int i, int64 batch_size) {
    TensorShape shape({batch_size});
    shape.AppendShape(component_shapes_[i]);
    return shape;
  }

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // Helper for dequeuing a single element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

<<<<<<< HEAD
  static Status GetElementComponentFromBatch(const Tuple& tuple, int64 index,
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

  mutex mu_;
  typedef std::deque<PersistentTensor> SubQueue;
  std::vector<SubQueue> queues_ GUARDED_BY(mu_);
  bool closed_ GUARDED_BY(mu_);

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

  static Status GetElementComponentFromBatch(const Tuple& tuple, int index,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                             int component,
                                             OpKernelContext* ctx,
                                             PersistentTensor* out_element);

<<<<<<< HEAD
 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FIFOQueue);
};

// Defines a FIFOQueueOp, which produces a Queue (specifically, one
// backed by FIFOQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
class FIFOQueueOp : public TypedQueueOp {
 public:
  explicit FIFOQueueOp(OpKernelConstruction* context);

 private:
  Status CreateResource(QueueInterface** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::vector<TensorShape> component_shapes_;
  TF_DISALLOW_COPY_AND_ASSIGN(FIFOQueueOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FIFO_QUEUE_H_
=======
  TF_DISALLOW_COPY_AND_ASSIGN(FIFOQueue);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FIFO_QUEUE_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
