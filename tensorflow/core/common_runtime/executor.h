#ifndef TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_
#define TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

class StepStatsCollector;

// Executor runs a graph computation.
// Example:
//   Graph* graph = ...;
//      ... construct graph ...
//   Executor* executor;
//   TF_CHECK_OK(NewSimpleExecutor(my_device, graph, &executor));
//   Rendezvous* rendezvous = NewNaiveRendezvous();
//   TF_CHECK_OK(rendezvous->Send("input", some_input_tensor));
//   TF_CHECK_OK(executor->Run({ExecutorOpts, rendezvous, nullptr}));
//   TF_CHECK_OK(rendezvous->Recv("input", &output_tensor));
//   ... ...
//
// Multiple threads can call Executor::Run concurrently.
class Executor {
 public:
  virtual ~Executor() {}

  // RunAsync() executes the graph computation. "done" is run when the
  // graph computation completes. If any error happens during the
  // computation, "done" is run and the error is passed to "done".
  //
  // RunAsync() is given a few arguments in Args. The caller must
  // ensure objects passed in Args (rendezvous, stats_collector, etc.)
  // are alive at least until done is invoked. All pointers to the
  // argument objects can be nullptr.
  //
  // RunAsync() uses the given "rendezvous", if not null, as the
  // mechanism to communicate inputs and outputs of the underlying
  // graph computation.
  //
  // RunAsync() calls "stats_collector", if not null, to keep track of
  // stats. This allows us to collect statistics and traces on demand.
  //
  // RunAsync() is provided a "call_frame", if the executor is used
  // for executing a function, is used to pass arguments and return
  // values between the caller and the callee.
  //
  // RunAsync() uses "cancellation_manager", if not nullptr, to
  // register callbacks that should be called if the graph computation
  // is cancelled. Note that the callbacks merely unblock any
  // long-running computation, and a cancelled step will terminate by
  // returning/calling the DoneCallback as usual.
  //
  // RunAsync() dispatches closures to "runner". Typically, "runner"
  // is backed up by a bounded threadpool.
  struct Args {
    Rendezvous* rendezvous = nullptr;
    StepStatsCollector* stats_collector = nullptr;
    FunctionCallFrame* call_frame = nullptr;
    CancellationManager* cancellation_manager = nullptr;

    typedef std::function<void()> Closure;
    typedef std::function<void(Closure)> Runner;
    Runner runner = nullptr;
  };
  typedef std::function<void(const Status&)> DoneCallback;
  virtual void RunAsync(const Args& args, DoneCallback done) = 0;

  // Synchronous wrapper for RunAsync().
  Status Run(const Args& args) {
    Status ret;
    Notification n;
    RunAsync(args, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
};

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor". The
// caller keeps the ownership of "device". The returned executor takes
// the ownership of "graph". Otherwise, returns an error status.
//
// "params" provides a set of context for the executor. We expect that
// different context would provide different implementations.
struct LocalExecutorParams {
  Device* device;

  // The library runtime support.
  FunctionLibraryRuntime* function_library;

  // True iff the computation contains control flow nodes.
  bool has_control_flow;

  // create_kernel returns an instance of op kernel based on NodeDef.
  // delete_kernel is called for every kernel used by the executor
  // when the executor is deleted.
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
  std::function<void(OpKernel*)> delete_kernel;
};
::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params,
                                      const Graph* graph, Executor** executor);

// A class to help run multiple executors in parallel and wait until
// all of them are complete.
//
// ExecutorBarrier deletes itself after the function returned by Get()
// is called.
class ExecutorBarrier {
 public:
  typedef std::function<void(const Status&)> StatusCallback;

  // Create an ExecutorBarrier for 'num' different executors.
  //
  // 'r' is the shared Rendezvous object that is used to communicate
  // state.  If any of the executors experiences an error, the
  // rendezvous object will be aborted exactly once.
  //
  // 'done' is called after the last executor completes, and
  // ExecutorBarrier is deleted.
  ExecutorBarrier(int num, Rendezvous* r, StatusCallback done)
      : rendez_(r), done_cb_(done), pending_(num) {}

  ~ExecutorBarrier() {}

  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  StatusCallback Get() {
    return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  }

 private:
  Rendezvous* rendez_ = nullptr;
  StatusCallback done_cb_ = nullptr;

  mutable mutex mu_;
  int pending_ GUARDED_BY(mu_) = 0;
  Status status_ GUARDED_BY(mu_);

  void WhenDone(const Status& s) {
    bool error = false;
    StatusCallback done = nullptr;
    Status status;
    {
      mutex_lock l(mu_);
      // If we are the first error encountered, mark the status
      // appropriately and later trigger an abort of the Rendezvous
      // object by this thread only.
      if (status_.ok() && !s.ok()) {
        error = true;
        status_ = s;
      }

      // If this is the last call to WhenDone, call the final callback
      // below.
      if (--pending_ == 0) {
        CHECK(done_cb_ != nullptr);
        done = done_cb_;
        done_cb_ = nullptr;
      }
      status = status_;
    }
    if (error) {
      rendez_->StartAbort(status);
    }
    if (done != nullptr) {
      delete this;
      done(status);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorBarrier);
};

// A few helpers to facilitate create/delete kernels.

// Creates a kernel based on "ndef" on device "device". The kernel can
// access the functions in the "flib". The caller takes ownership of
// returned "*kernel".
Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, OpKernel** kernel);

// Deletes "kernel" returned by CreateKernel.
void DeleteNonCachedKernel(OpKernel* kernel);

// Creates a kernel based on "ndef" on device "device". The kernel can
// access the functions in the "flib". The caller does not take
// ownership of returned "*kernel". If a kernel has been created for
// ndef.name(), returns the same kernel instance.
Status CreateCachedKernel(Device* device, const string& session,
                          FunctionLibraryRuntime* flib, const NodeDef& ndef,
                          OpKernel** kernel);

// Deletes "kernel" returned by CreateCachedKernel.
void DeleteCachedKernel(Device* device, const string& session,
                        OpKernel* kernel);

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_
