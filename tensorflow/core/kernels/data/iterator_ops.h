/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_ITERATOR_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_ITERATOR_OPS_H_

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_util.h"

namespace tensorflow {
namespace data {

class IteratorResource;

class IteratorHandleOp : public OpKernel {
 public:
  explicit IteratorHandleOp(OpKernelConstruction* ctx);

  // The resource is deleted from the resource manager only when it is private
  // to kernel. Ideally the resource should be deleted when it is no longer held
  // by anyone, but it would break backward compatibility.
  ~IteratorHandleOp() override;

  void Compute(OpKernelContext* context) override LOCKS_EXCLUDED(mu_);

 private:
  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  Status VerifyResource(IteratorResource* resource);

  template <typename To, typename From>  // use like this: down_cast<T*>(foo);
  static inline To down_cast(From* f) {  // so we only accept pointers
    static_assert(
        (std::is_base_of<From, typename std::remove_pointer<To>::type>::value),
        "target type not derived from source type");

    // We skip the assert and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
    // Uses RTTI in dbg and fastbuild. asserts are disabled in opt builds.
    assert(f == nullptr || dynamic_cast<To>(f) != nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)
    return static_cast<To>(f);
  }

  FunctionLibraryRuntime* CreatePrivateFLR(
      OpKernelContext* ctx, std::unique_ptr<DeviceMgr>* device_mgr,
      std::unique_ptr<FunctionLibraryDefinition>* flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime>* pflr);

  mutex mu_;
  ContainerInfo cinfo_;  // Written once under mu_ then constant afterwards.
  IteratorResource* resource_ GUARDED_BY(mu_) = nullptr;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
  string name_;
};

template <typename T>
class AnonymousIteratorResourceOp : public OpKernel {
 public:
  explicit AnonymousIteratorResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("output_types", &output_dtypes_));
    OP_REQUIRES_OK(context, context->GetAttr("output_shapes", &output_shapes_));
  }

  void Compute(OpKernelContext* ctx) override {
    FunctionLibraryRuntime* lib;
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
    OP_REQUIRES_OK(
        ctx, ctx->function_library()->Clone(&flib_def, &pflr, &lib, true));
    T* resource;
    OP_REQUIRES_OK(ctx, CreateResource(ctx, std::move(flib_def),
                                       std::move(pflr), lib, &resource));

    string unique_name, container_name;
    GenerateContainerNames(&unique_name, &container_name);
    ResourceMgr* mgr = ctx->resource_manager();
    OP_REQUIRES_OK(ctx, mgr->Create<T>(container_name, unique_name, resource));

    Tensor* handle_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle_t));
    ResourceHandle handle = MakeResourceHandle(ctx, container_name, unique_name,
                                               MakeTypeIndex<T>());
    handle_t->scalar<ResourceHandle>()() = handle;

    if (create_deleter_) {
      Tensor* deleter_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &deleter_t));
      deleter_t->scalar<Variant>()() =
          ResourceDeleter(handle, ctx->resource_manager());
    }
  }

 protected:
  virtual void GenerateContainerNames(string* unique_name,
                                      string* container_name) = 0;

  virtual Status CreateResource(
      OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
      FunctionLibraryRuntime* lib, T** resource) = 0;

  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  bool create_deleter_ = true;
};

// Like IteratorHandleOp, but creates handles which are never shared, and does
// not hold a reference to these handles. The latter is important for eager
// execution, since OpKernel instances generally live as long as the program
// running them.
class AnonymousIteratorHandleOp
    : public AnonymousIteratorResourceOp<IteratorResource> {
 public:
  explicit AnonymousIteratorHandleOp(OpKernelConstruction* context);

 private:
  void GenerateContainerNames(string* unique_name,
                              string* container_name) override;

  Status CreateResource(OpKernelContext* ctx,
                        std::unique_ptr<FunctionLibraryDefinition> flib_def,
                        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                        FunctionLibraryRuntime* lib,
                        IteratorResource** resource) override;

  const int graph_def_version_;
};

class MakeIteratorOp : public AsyncOpKernel {
 public:
  explicit MakeIteratorOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "tf_data_make_iterator") {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  BackgroundWorker background_worker_;
};

class IteratorGetNextOp : public AsyncOpKernel {
 public:
  explicit IteratorGetNextOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "tf_data_iterator_get_next") {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  BackgroundWorker background_worker_;
};

class DeleteIteratorOp : public OpKernel {
 public:
  explicit DeleteIteratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class IteratorGetNextAsOptionalOp : public AsyncOpKernel {
 public:
  explicit IteratorGetNextAsOptionalOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(),
                           "tf_data_iterator_get_next_as_optional") {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  BackgroundWorker background_worker_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class IteratorGetNextSyncOp : public OpKernel {
 public:
  explicit IteratorGetNextSyncOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class IteratorToStringHandleOp : public OpKernel {
 public:
  explicit IteratorToStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class IteratorFromStringHandleOp : public OpKernel {
 public:
  explicit IteratorFromStringHandleOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_ITERATOR_OPS_H_
