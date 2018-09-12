/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/captured_function.h"

#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/notification.h"

namespace tensorflow {
namespace data {

/* static */
Status CapturedFunction::Create(
    const NameAttrList& func, std::vector<Tensor> captured_inputs,
    std::unique_ptr<CapturedFunction>* out_function) {
  return Create(func, std::move(captured_inputs), true, out_function);
}

/* static */
Status CapturedFunction::Create(
    const NameAttrList& func, std::vector<Tensor> captured_inputs,
    bool use_inter_op_parallelism,
    std::unique_ptr<CapturedFunction>* out_function) {
  out_function->reset(new CapturedFunction(func, std::move(captured_inputs),
                                           use_inter_op_parallelism));
  return Status::OK();
}

/* static */
Status CapturedFunction::Create(
    const NameAttrList& func, OpKernelContext* ctx, const string& argument,
    std::unique_ptr<CapturedFunction>* out_function) {
  OpInputList argument_inputs;
  TF_RETURN_IF_ERROR(ctx->input_list(argument, &argument_inputs));
  std::vector<Tensor> arguments_t;
  arguments_t.reserve(argument_inputs.size());
  for (const Tensor& t : argument_inputs) {
    arguments_t.push_back(t);
  }
  return CapturedFunction::Create(func, std::move(arguments_t), out_function);
}

CapturedFunction::~CapturedFunction() {
  if (lib_ != nullptr && f_handle_ != kInvalidHandle) {
    lib_->ReleaseHandle(f_handle_).IgnoreError();
  }
}

namespace {
class CallFrameBase : public CallFrameInterface {
 public:
  explicit CallFrameBase(DataTypeSlice ret_types)
      : ret_types_(ret_types), retvals_(ret_types.size()) {}

  // Caller methods.
  Status ConsumeRetvals(std::vector<Tensor>* retvals) {
    retvals->reserve(retvals_.size());
    int i = 0;
    for (auto&& val : retvals_) {
      if (!val) {
        return errors::Internal("No return value for index ", i, ".");
      }
      retvals->emplace_back(std::move(val.value()));
      ++i;
    }
    return Status::OK();
  }

  size_t num_retvals() const override { return retvals_.size(); }

  // Callee methods.
  Status SetRetval(int index, const Tensor& val) override {
    if (index < retvals_.size() && val.dtype() == ret_types_[index] &&
        !retvals_[index]) {
      retvals_[index] = val;
      return Status::OK();
    } else if (index >= retvals_.size()) {
      return errors::InvalidArgument("Return value ", index,
                                     " is out of range.");
    } else if (val.dtype() != ret_types_[index]) {
      return errors::InvalidArgument("Expected type ",
                                     DataTypeString(ret_types_[index]),
                                     " for return value ", index, " but got ",
                                     DataTypeString(val.dtype()), ".");
    } else {
      return errors::Internal("Attempted to set return value ", index,
                              " more than once.");
    }
  }

 private:
  DataTypeSlice ret_types_;
  std::vector<gtl::optional<Tensor>> retvals_;
  TF_DISALLOW_COPY_AND_ASSIGN(CallFrameBase);
};

class OwnedArgsCallFrame : public CallFrameBase {
 public:
  OwnedArgsCallFrame(std::vector<Tensor>&& args,
                     const std::vector<Tensor>* captured_inputs,
                     DataTypeSlice ret_types)
      : CallFrameBase(ret_types),
        args_(std::move(args)),
        captured_inputs_(captured_inputs) {}

  size_t num_args() const override {
    return args_.size() + captured_inputs_->size();
  }

  // Callee methods.
  Status GetArg(int index, Tensor* val) const override {
    if (index < args_.size() && args_[index].IsInitialized()) {
      // TODO(mrry): Consider making `CallFrameInterface::GetArg` non-const in
      // order to be able to `std::move(args_[index])` into `*val`.
      *val = args_[index];
      return Status::OK();
    } else if (index < args_.size() + captured_inputs_->size()) {
      *val = (*captured_inputs_)[index - args_.size()];
      return Status::OK();
    } else if (index >= args_.size() + captured_inputs_->size()) {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
    } else {
      return errors::Internal("Attempted to get argument ", index,
                              " more than once.");
    }
  }

 private:
  std::vector<Tensor> args_;
  const std::vector<Tensor>* const captured_inputs_;  // Not owned.
};

class BorrowedArgsCallFrame : public CallFrameBase {
 public:
  BorrowedArgsCallFrame(const std::vector<Tensor>& args,
                        const std::vector<Tensor>* captured_inputs,
                        DataTypeSlice ret_types)
      : CallFrameBase(ret_types),
        args_(args),
        captured_inputs_(captured_inputs) {}

  size_t num_args() const override {
    return args_.size() + captured_inputs_->size();
  }

  // Callee methods.
  Status GetArg(int index, Tensor* val) const override {
    if (index < args_.size() && args_[index].IsInitialized()) {
      *val = args_[index];
      return Status::OK();
    } else if (index < args_.size() + captured_inputs_->size()) {
      *val = (*captured_inputs_)[index - args_.size()];
      return Status::OK();
    } else if (index >= args_.size() + captured_inputs_->size()) {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
    } else {
      return errors::Internal("Attempted to get argument ", index,
                              " more than once.");
    }
  }

 private:
  const std::vector<Tensor>& args_;                   // Not owned.
  const std::vector<Tensor>* const captured_inputs_;  // Not owned.
};

}  // namespace

Status CapturedFunction::GetHandle(IteratorContext* ctx,
                                   FunctionLibraryRuntime::Handle* out_handle) {
  tf_shared_lock l(mu_);
  if (lib_ == nullptr) {
    return errors::Internal("Captured function \"", func_.name(),
                            "\" was called before it was instantiated.");
  }
  if (ctx->lib() != lib_) {
    return errors::Internal("Captured function \"", func_.name(),
                            "\" was called with a different "
                            "FunctionLibraryRuntime*, which is not permitted.");
  }
  *out_handle = f_handle_;
  return Status::OK();
}

Status CapturedFunction::Run(IteratorContext* ctx, std::vector<Tensor>&& args,
                             std::vector<Tensor>* rets) {
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(GetHandle(ctx, &handle));

  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = CapturedFunction::generate_step_id();
  ScopedStepContainer step_container(f_opts.step_id, [ctx](const string& name) {
    ctx->lib()->device()->resource_manager()->Cleanup(name).IgnoreError();
  });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  if (ctx->lib()->device()->device_type() != DEVICE_CPU) {
    f_opts.create_rendezvous = true;
  }
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  CancellationManager c_mgr;
  f_opts.cancellation_manager = &c_mgr;

  OwnedArgsCallFrame frame(std::move(args), &captured_inputs_, ret_types_);
  Notification n;
  Status s;
  ctx->lib()->Run(f_opts, handle, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

Status CapturedFunction::RunWithBorrowedArgs(IteratorContext* ctx,
                                             const std::vector<Tensor>& args,
                                             std::vector<Tensor>* rets) {
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(GetHandle(ctx, &handle));

  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = CapturedFunction::generate_step_id();
  ScopedStepContainer step_container(f_opts.step_id, [ctx](const string& name) {
    ctx->lib()->device()->resource_manager()->Cleanup(name).IgnoreError();
  });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  if (ctx->lib()->device()->device_type() != DEVICE_CPU) {
    f_opts.create_rendezvous = true;
  }
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  CancellationManager c_mgr;
  f_opts.cancellation_manager = &c_mgr;

  BorrowedArgsCallFrame frame(args, &captured_inputs_, ret_types_);
  Notification n;
  Status s;

  ctx->lib()->Run(f_opts, handle, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

Status CapturedFunction::Instantiate(IteratorContext* ctx) {
  mutex_lock l(mu_);
  if (lib_ == nullptr) {
    // The context's runtime will be used for all subsequent calls.
    lib_ = ctx->lib();
    DCHECK(f_handle_ == kInvalidHandle);
    FunctionLibraryRuntime::InstantiateOptions inst_opts;
    inst_opts.overlay_lib = ctx->function_library().get();
    inst_opts.state_handle = std::to_string(random::New64());
    inst_opts.create_kernels_eagerly = true;
    if (!use_inter_op_parallelism_) {
      inst_opts.executor_type = "SINGLE_THREADED_EXECUTOR";
    }
    Status s = (lib_->Instantiate(func_.name(), AttrSlice(&func_.attr()),
                                  inst_opts, &f_handle_));
    TF_RETURN_IF_ERROR(s);
    const FunctionBody* fbody = lib_->GetFunctionBody(f_handle_);
    if (fbody == nullptr) {
      return errors::Internal("Failed to instantiate function body.");
    }
    ret_types_ = fbody->ret_types;
  } else {
    if (ctx->lib() != lib_) {
      return errors::Internal(
          "Captured function was called with a different "
          "FunctionLibraryRuntime*, which is not permitted.");
    }
  }
  if (captured_runner_ == nullptr) {
    captured_runner_ = *ctx->runner();
  }
  return Status::OK();
}

Status CapturedFunction::RunInstantiated(const std::vector<Tensor>& args,
                                         std::vector<Tensor>* rets) {
  FunctionLibraryRuntime* lib;
  FunctionLibraryRuntime::Handle handle;
  std::function<void(std::function<void()>)>* runner;
  {
    tf_shared_lock l(mu_);
    if (lib_ == nullptr) {
      return errors::FailedPrecondition(
          "`CapturedFunction::Instantiate()` must be called before a call to "
          "`CapturedFunction::RunInstantiated()`.");
    }
    lib = lib_;
    handle = f_handle_;
    runner = &captured_runner_;
  }

  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = CapturedFunction::generate_step_id();
  ScopedStepContainer step_container(f_opts.step_id, [lib](const string& name) {
    lib->device()->resource_manager()->Cleanup(name).IgnoreError();
  });
  f_opts.step_container = &step_container;
  f_opts.runner = runner;
  if (lib->device()->device_type() != DEVICE_CPU) {
    f_opts.create_rendezvous = true;
  }
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  CancellationManager c_mgr;
  f_opts.cancellation_manager = &c_mgr;

  BorrowedArgsCallFrame frame(args, &captured_inputs_, ret_types_);
  Notification n;
  Status s;

  lib->Run(f_opts, handle, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

void CapturedFunction::RunAsync(IteratorContext* ctx,
                                std::vector<Tensor>&& args,
                                std::vector<Tensor>* rets,
                                FunctionLibraryRuntime::DoneCallback done,
                                const string& prefix) {
  // NOTE(mrry): This method does not transfer ownership of `ctx`, and it may
  // be deleted before `done` is called. Take care not to capture `ctx` in any
  // code that may execute asynchronously in this function.
  FunctionLibraryRuntime::Handle handle;
  Status s = GetHandle(ctx, &handle);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto frame =
      new OwnedArgsCallFrame(std::move(args), &captured_inputs_, ret_types_);

  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = CapturedFunction::generate_step_id();
  ResourceMgr* resource_mgr = ctx->lib()->device()->resource_manager();
  auto step_container = new ScopedStepContainer(
      f_opts.step_id, [resource_mgr](const string& name) {
        resource_mgr->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = step_container;
  f_opts.runner = ctx->runner();
  if (ctx->lib()->device()->device_type() != DEVICE_CPU) {
    f_opts.create_rendezvous = true;
  }
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  auto c_mgr = new CancellationManager;
  f_opts.cancellation_manager = c_mgr;
  StepStats* stats = nullptr;
  StepStatsCollector* stats_collector = nullptr;
  std::shared_ptr<model::Node> node;
  if (ctx->model()) {
    node = ctx->model()->LookupNode(prefix);
    if (node) {
      // TODO(b/114104975): Use something light-weight here.
      stats = new StepStats();
      stats_collector = new StepStatsCollector(stats);
    }
  }
  f_opts.stats_collector = stats_collector;

  auto callback = std::bind(
      [rets, step_container, c_mgr, frame, stats, stats_collector, node](
          FunctionLibraryRuntime::DoneCallback done,
          // Begin unbound arguments.
          Status s) {
        delete step_container;
        delete c_mgr;
        if (s.ok()) {
          s = frame->ConsumeRetvals(rets);
        }
        delete frame;
        if (node) {
          int64 delta = 0;
          stats_collector->Finalize();
          for (auto dev_stats : stats->dev_stats()) {
            for (auto node_stats : dev_stats.node_stats()) {
              delta += node_stats.all_end_rel_nanos();
            }
          }
          delete stats_collector;
          delete stats;
          node->add_processing_time(delta);
          node->start_work();
        }
        done(s);
        if (node) {
          node->stop_work();
        }
      },
      std::move(done), std::placeholders::_1);

  ctx->lib()->Run(f_opts, handle, frame, std::move(callback));
}

CapturedFunction::CapturedFunction(const NameAttrList& func,
                                   std::vector<Tensor> captured_inputs,
                                   bool use_inter_op_parallelism)
    : func_(func),
      lib_(nullptr),
      f_handle_(kInvalidHandle),
      captured_inputs_(std::move(captured_inputs)),
      use_inter_op_parallelism_(use_inter_op_parallelism) {}

}  // namespace data
}  // namespace tensorflow
