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
#include "tensorflow/core/framework/dataset.h"

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

namespace {

// A wrapper class for storing a `DatasetBase` instance in a DT_VARIANT tensor.
// Objects of the wrapper class own a reference on an instance of `DatasetBase`,
// and the wrapper's copy constructor and destructor take care of managing the
// reference count.
//
// NOTE(mrry): This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot currently serialize an arbitrary
// `DatasetBase` object, so the `Encode()` and `Decode()` methods are not
// implemented.
class DatasetVariantWrapper {
 public:
  DatasetVariantWrapper() : dataset_(nullptr) {}

  // Transfers ownership of `dataset` to `*this`.
  explicit DatasetVariantWrapper(DatasetBase* dataset) : dataset_(dataset) {}

  DatasetVariantWrapper(const DatasetVariantWrapper& other)
      : dataset_(other.dataset_) {
    if (dataset_) dataset_->Ref();
  }

  ~DatasetVariantWrapper() {
    if (dataset_) dataset_->Unref();
  }

  DatasetBase* get() const { return dataset_; }

  string TypeName() const { return "tensorflow::DatasetVariantWrapper"; }
  string DebugString() const {
    if (dataset_) {
      return dataset_->DebugString();
    } else {
      return "<Uninitialized DatasetVariantWrapper>";
    }
  }
  void Encode(VariantTensorData* data) const {
    LOG(ERROR) << "The Encode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
  }
  bool Decode(const VariantTensorData& data) {
    LOG(ERROR) << "The Decode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
    return false;
  }

 private:
  DatasetBase* const dataset_;  // Owns one reference.
};

}  // namespace

Status GraphDefBuilderWrapper::AddDataset(
    const GraphDatasetBase* dataset,
    const std::vector<std::pair<size_t, Node*>>& inputs,
    const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
    Node** output) {
  const string& op_type_name = dataset->op_name();
  std::unique_ptr<const GraphDefBuilder::Options> opts(
      new GraphDefBuilder::Options(b_->opts()));
  // TODO(srbs|mrry): Not all datasets have output_types and output_shapes
  // attributes defined. It will be nice to have a consistent pattern.
  bool has_output_types_attr = HasAttr(op_type_name, "output_types");
  bool has_output_shapes_attr = HasAttr(op_type_name, "output_shapes");
  if (has_output_shapes_attr) {
    opts.reset(new GraphDefBuilder::Options(
        opts->WithAttr("output_shapes", dataset->output_shapes())));
  }
  if (has_output_types_attr) {
    opts.reset(new GraphDefBuilder::Options(
        opts->WithAttr("output_types", dataset->output_dtypes())));
  }
  for (auto attr : attrs) {
    opts.reset(
        new GraphDefBuilder::Options(opts->WithAttr(attr.first, attr.second)));
  }
  if (opts->HaveError()) {
    return errors::Internal("AddDataset: Failed to build Options with error ",
                            opts->StatusToString());
  }
  NodeBuilder node_builder(opts->GetNameForOp(op_type_name), op_type_name,
                           opts->op_registry());
  {
    size_t total_size = inputs.size() + list_inputs.size();
    auto inputs_iter = inputs.begin();
    auto list_inputs_iter = list_inputs.begin();
    for (int i = 0; i < total_size; i++) {
      if (inputs_iter != inputs.end() && inputs_iter->first == i) {
        node_builder.Input(NodeBuilder::NodeOut(inputs_iter->second));
        inputs_iter++;
      } else if (list_inputs_iter != list_inputs.end() &&
                 list_inputs_iter->first == i) {
        std::vector<NodeBuilder::NodeOut> nodeout_inputs;
        nodeout_inputs.reserve(list_inputs_iter->second.size());
        for (Node* n : list_inputs_iter->second) {
          nodeout_inputs.emplace_back(n);
        }
        node_builder.Input(nodeout_inputs);
        list_inputs_iter++;
      } else {
        return errors::InvalidArgument("No input found for index ", i);
      }
    }
  }
  *output = opts->FinalizeBuilder(&node_builder);
  if (*output == nullptr) {
    return errors::Internal("AddDataset: Failed to build ", op_type_name,
                            " op with error ", opts->StatusToString());
  }
  return Status::OK();
}

Status GraphDefBuilderWrapper::AddFunction(
    const FunctionLibraryDefinition& flib_def, const string& function_name) {
  if (b_->HasFunction(function_name)) {
    VLOG(1) << "Function with name " << function_name << "already exists in"
            << " the graph. It will not be added again.";
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(EnsureFunctionIsStateless(flib_def, function_name));
  const FunctionDef* f_def = flib_def.Find(function_name);
  if (f_def == nullptr) {
    return errors::InvalidArgument("Unable to find FunctionDef for ",
                                   function_name, " in the registry.");
  }
  FunctionDefLibrary def;
  *def.add_function() = *f_def;
  const string gradient_func = flib_def.FindGradient(function_name);
  if (!gradient_func.empty()) {
    GradientDef* g_def = def.add_gradient();
    g_def->set_function_name(function_name);
    g_def->set_gradient_func(gradient_func);
  }
  TF_RETURN_IF_ERROR(b_->AddFunctionLibrary(def));

  // Recursively add functions in inputs of function_name.
  for (const NodeDef& node_def : f_def->node_def()) {
    const OpRegistrationData* op_reg_data = nullptr;
    TF_RETURN_IF_ERROR(flib_def.LookUp(node_def.op(), &op_reg_data));
    if (op_reg_data->is_function_op) {
      TF_RETURN_IF_ERROR(AddFunction(flib_def, op_reg_data->op_def.name()));
    }
    // Recursively add functions in attrs of this NodeDef.
    for (const auto& pair : node_def.attr()) {
      TF_RETURN_IF_ERROR(AddAttrFunctions(pair.second, flib_def));
    }
  }

  // Recursively add functions in attrs of function_name.
  for (auto iter = f_def->attr().begin(); iter != f_def->attr().end(); iter++) {
    TF_RETURN_IF_ERROR(AddAttrFunctions(iter->second, flib_def));
  }
  return Status::OK();
}

void GraphDefBuilderWrapper::AddTensorInternal(const Tensor& val,
                                               Node** output) {
  *output = ops::SourceOp(
      "Const",
      b_->opts().WithAttr("dtype", val.dtype()).WithAttr("value", val));
}

bool GraphDefBuilderWrapper::HasAttr(const string& op_type_name,
                                     const string& attr_name) const {
  const OpDef* op_def = nullptr;
  Status s = b_->opts().op_registry()->LookUpOpDef(op_type_name, &op_def);
  if (!s.ok() || op_def == nullptr) {
    return false;
  }
  return HasAttr(op_def, attr_name);
}

Status GraphDatasetBase::Serialize(SerializationContext* ctx,
                                   string* serialized_graph_def,
                                   string* output_node) const {
  GraphDefBuilder b;
  DatasetGraphDefBuilder db(&b);
  Node* node = nullptr;
  TF_RETURN_IF_ERROR(AsGraphDefInternal(ctx, &db, &node));
  *output_node = node->name();
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(b.ToGraphDef(&graph_def));
  graph_def.SerializeToString(serialized_graph_def);
  return Status::OK();
}

Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset) {
  if (!(tensor.dtype() == DT_VARIANT ||
        TensorShapeUtils::IsScalar(tensor.shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  const Variant& variant = tensor.scalar<Variant>()();
  const DatasetVariantWrapper* wrapper = variant.get<DatasetVariantWrapper>();
  if (wrapper == nullptr) {
    return errors::InvalidArgument("Tensor must be a Dataset object.");
  }
  *out_dataset = wrapper->get();
  if (*out_dataset == nullptr) {
    return errors::Internal("Read uninitialized Dataset variant.");
  }
  return Status::OK();
}

Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor) {
  if (!(tensor->dtype() == DT_VARIANT ||
        TensorShapeUtils::IsScalar(tensor->shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  tensor->scalar<Variant>()() = DatasetVariantWrapper(dataset);
  return Status::OK();
}

void DatasetOpKernel::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset = nullptr;
  MakeDataset(ctx, &dataset);
  if (ctx->status().ok()) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(ctx, StoreDatasetInVariantTensor(dataset, output));
  }
}

void UnaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  MakeDataset(ctx, input, output);
}

void BinaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                        DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  DatasetBase* another_input;
  OP_REQUIRES_OK(ctx,
                 GetDatasetFromVariantTensor(ctx->input(1), &another_input));
  MakeDataset(ctx, input, another_input, output);
}

const char GraphDatasetBase::kDatasetGraphKey[] = "_DATASET_GRAPH";
const char GraphDatasetBase::kDatasetGraphOutputNodeKey[] =
    "_DATASET_GRAPH_OUTPUT_NODE";

BackgroundWorker::BackgroundWorker(Env* env, const string& name) {
  thread_.reset(env->StartThread({} /* thread_options */, name,
                                 [this]() { WorkerLoop(); }));
}

BackgroundWorker::~BackgroundWorker() {
  {
    mutex_lock l(mu_);
    cancelled_ = true;
  }
  cond_var_.notify_one();
  // Block until the background thread has terminated.
  //
  // NOTE(mrry): We explicitly free and join the thread here because
  // `WorkerLoop()` uses other members of this object, and so we must join
  // the thread before destroying them.
  thread_.reset();
}

void BackgroundWorker::Schedule(std::function<void()> work_item) {
  {
    mutex_lock l(mu_);
    work_queue_.push_back(std::move(work_item));
  }
  cond_var_.notify_one();
}

void BackgroundWorker::WorkerLoop() {
  while (true) {
    std::function<void()> work_item = nullptr;
    {
      mutex_lock l(mu_);
      while (!cancelled_ && work_queue_.empty()) {
        cond_var_.wait(l);
      }
      if (cancelled_) {
        return;
      }
      DCHECK(!work_queue_.empty());
      work_item = std::move(work_queue_.front());
      work_queue_.pop_front();
    }
    DCHECK(work_item != nullptr);
    work_item();
  }
}

namespace dataset {

IteratorContext MakeIteratorContext(OpKernelContext* ctx) {
  IteratorContext::Params params;
  params.env = ctx->env();
  params.runner = *(ctx->runner());
  params.lib = ctx->function_library();
  // Note: must use reinterpret_cast because function.h forward-declares Device.
  DeviceBase* device =
      reinterpret_cast<DeviceBase*>(ctx->function_library()->device());
  params.allocator_getter = [device](AllocatorAttributes attrs) {
    return device->GetAllocator(attrs);
  };
  return IteratorContext(params);
}

}  // namespace dataset

}  // namespace tensorflow
