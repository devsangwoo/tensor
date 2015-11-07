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
#include "tensorflow/core/framework/op.h"

#include <algorithm>
#include <memory>
#include <vector>
<<<<<<< HEAD

#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
=======
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/protobuf.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

// OpRegistry -----------------------------------------------------------------

OpRegistryInterface::~OpRegistryInterface() {}

<<<<<<< HEAD
Status OpRegistryInterface::LookUpOpDef(const string& op_type_name,
                                        const OpDef** op_def) const {
  *op_def = nullptr;
  const OpRegistrationData* op_reg_data = nullptr;
  TF_RETURN_IF_ERROR(LookUp(op_type_name, &op_reg_data));
  *op_def = &op_reg_data->op_def;
  return Status::OK();
}

OpRegistry::OpRegistry() : initialized_(false) {}

OpRegistry::~OpRegistry() {
  for (const auto& e : registry_) delete e.second;
}

void OpRegistry::Register(const OpRegistrationDataFactory& op_data_factory) {
  mutex_lock lock(mu_);
  if (initialized_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_data_factory));
  } else {
    deferred_.push_back(op_data_factory);
  }
}

namespace {
// Helper function that returns Status message for failed LookUp.
Status OpNotFound(const string& op_type_name) {
  Status status = errors::NotFound(
      "Op type not registered '", op_type_name, "' in binary running on ",
      port::Hostname(), ". ",
      "Make sure the Op and Kernel are registered in the binary running in "
      "this process. Note that if you are loading a saved graph which used ops "
      "from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done "
      "before importing the graph, as contrib ops are lazily registered when "
      "the module is first accessed.");
  VLOG(1) << status.ToString();
  return status;
}
}  // namespace

Status OpRegistry::LookUp(const string& op_type_name,
                          const OpRegistrationData** op_reg_data) const {
  if ((*op_reg_data = LookUp(op_type_name))) return Status::OK();
  return OpNotFound(op_type_name);
}

const OpRegistrationData* OpRegistry::LookUp(const string& op_type_name) const {
  {
    tf_shared_lock l(mu_);
    if (initialized_) {
      if (const OpRegistrationData* res =
              gtl::FindWithDefault(registry_, op_type_name, nullptr)) {
        return res;
      }
    }
  }
  return LookUpSlow(op_type_name);
}

const OpRegistrationData* OpRegistry::LookUpSlow(
    const string& op_type_name) const {
  const OpRegistrationData* res = nullptr;

  bool first_call = false;
  bool first_unregistered = false;
  {  // Scope for lock.
    mutex_lock lock(mu_);
    first_call = MustCallDeferred();
    res = gtl::FindWithDefault(registry_, op_type_name, nullptr);

    static bool unregistered_before = false;
    first_unregistered = !unregistered_before && (res == nullptr);
    if (first_unregistered) {
      unregistered_before = true;
    }
    // Note: Can't hold mu_ while calling Export() below.
  }
  if (first_call) {
    TF_QCHECK_OK(ValidateKernelRegistrations(*this));
  }
  if (res == nullptr) {
    if (first_unregistered) {
      OpList op_list;
      Export(true, &op_list);
      if (VLOG_IS_ON(3)) {
        LOG(INFO) << "All registered Ops:";
        for (const auto& op : op_list.op()) {
          LOG(INFO) << SummarizeOpDef(op);
        }
      }
    }
  }
  return res;
}

void OpRegistry::GetRegisteredOps(std::vector<OpDef>* op_defs) {
  mutex_lock lock(mu_);
  MustCallDeferred();
  for (const auto& p : registry_) {
    op_defs->push_back(p.second->op_def);
  }
}

void OpRegistry::GetOpRegistrationData(
    std::vector<OpRegistrationData>* op_data) {
  mutex_lock lock(mu_);
  MustCallDeferred();
  for (const auto& p : registry_) {
    op_data->push_back(*p.second);
  }
}

Status OpRegistry::SetWatcher(const Watcher& watcher) {
  mutex_lock lock(mu_);
  if (watcher_ && watcher) {
    return errors::AlreadyExists(
        "Cannot over-write a valid watcher with another.");
  }
  watcher_ = watcher;
  return Status::OK();
=======
OpRegistry::OpRegistry() : initialized_(false) {}

void OpRegistry::Register(std::function<OpDef(void)> func) {
  mutex_lock lock(mu_);
  if (initialized_) {
    OpDef def = func();
    TF_QCHECK_OK(RegisterAlreadyLocked(def)) << "Attempting to register: "
                                             << SummarizeOpDef(def);
  } else {
    deferred_.push_back(func);
  }
}

const OpDef* OpRegistry::LookUp(const string& op_type_name,
                                Status* status) const {
  const OpDef* op_def = nullptr;
  bool first_call = false;
  {  // Scope for lock.
    mutex_lock lock(mu_);
    first_call = CallDeferred();
    op_def = gtl::FindWithDefault(registry_, op_type_name, nullptr);
    // Note: Can't hold mu_ while calling Export() below.
  }
  if (first_call) {
    TF_QCHECK_OK(ValidateKernelRegistrations(this));
  }
  if (op_def == nullptr) {
    status->Update(
        errors::NotFound("Op type not registered '", op_type_name, "'"));
    static bool first = true;
    if (first) {
      OpList op_list;
      Export(true, &op_list);
      LOG(INFO) << "All registered Ops:";
      for (const auto& op : op_list.op()) {
        LOG(INFO) << SummarizeOpDef(op);
      }
      first = false;
    }
  }
  return op_def;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

void OpRegistry::Export(bool include_internal, OpList* ops) const {
  mutex_lock lock(mu_);
<<<<<<< HEAD
  MustCallDeferred();

  std::vector<std::pair<string, const OpRegistrationData*>> sorted(
      registry_.begin(), registry_.end());
=======
  CallDeferred();

  std::vector<std::pair<string, const OpDef*>> sorted(registry_.begin(),
                                                      registry_.end());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  std::sort(sorted.begin(), sorted.end());

  auto out = ops->mutable_op();
  out->Clear();
  out->Reserve(sorted.size());

  for (const auto& item : sorted) {
<<<<<<< HEAD
    if (include_internal || !absl::StartsWith(item.first, "_")) {
      *out->Add() = item.second->op_def;
=======
    if (include_internal || !StringPiece(item.first).starts_with("_")) {
      *out->Add() = *item.second;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    }
  }
}

<<<<<<< HEAD
void OpRegistry::DeferRegistrations() {
  mutex_lock lock(mu_);
  initialized_ = false;
}

void OpRegistry::ClearDeferredRegistrations() {
  mutex_lock lock(mu_);
  deferred_.clear();
}

Status OpRegistry::ProcessRegistrations() const {
  mutex_lock lock(mu_);
  return CallDeferred();
}

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
string OpRegistry::DebugString(bool include_internal) const {
  OpList op_list;
  Export(include_internal, &op_list);
  string ret;
  for (const auto& op : op_list.op()) {
    strings::StrAppend(&ret, SummarizeOpDef(op), "\n");
  }
  return ret;
}

<<<<<<< HEAD
bool OpRegistry::MustCallDeferred() const {
  if (initialized_) return false;
  initialized_ = true;
  for (size_t i = 0; i < deferred_.size(); ++i) {
    TF_QCHECK_OK(RegisterAlreadyLocked(deferred_[i]));
=======
bool OpRegistry::CallDeferred() const {
  if (initialized_) return false;
  initialized_ = true;
  for (const auto& fn : deferred_) {
    OpDef def = fn();
    TF_QCHECK_OK(RegisterAlreadyLocked(def)) << "Attempting to register: "
                                             << SummarizeOpDef(def);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
  deferred_.clear();
  return true;
}

<<<<<<< HEAD
Status OpRegistry::CallDeferred() const {
  if (initialized_) return Status::OK();
  initialized_ = true;
  for (size_t i = 0; i < deferred_.size(); ++i) {
    Status s = RegisterAlreadyLocked(deferred_[i]);
    if (!s.ok()) {
      return s;
    }
  }
  deferred_.clear();
  return Status::OK();
}

Status OpRegistry::RegisterAlreadyLocked(
    const OpRegistrationDataFactory& op_data_factory) const {
  std::unique_ptr<OpRegistrationData> op_reg_data(new OpRegistrationData);
  Status s = op_data_factory(op_reg_data.get());
  if (s.ok()) {
    s = ValidateOpDef(op_reg_data->op_def);
    if (s.ok() &&
        !gtl::InsertIfNotPresent(&registry_, op_reg_data->op_def.name(),
                                 op_reg_data.get())) {
      s = errors::AlreadyExists("Op with name ", op_reg_data->op_def.name());
    }
  }
  Status watcher_status = s;
  if (watcher_) {
    watcher_status = watcher_(s, op_reg_data->op_def);
  }
  if (s.ok()) {
    op_reg_data.release();
  } else {
    op_reg_data.reset();
  }
  return watcher_status;
=======
Status OpRegistry::RegisterAlreadyLocked(const OpDef& def) const {
  TF_RETURN_IF_ERROR(ValidateOpDef(def));

  std::unique_ptr<OpDef> copy(new OpDef(def));
  if (gtl::InsertIfNotPresent(&registry_, def.name(), copy.get())) {
    copy.release();  // Ownership transferred to op_registry
    return Status::OK();
  } else {
    return errors::AlreadyExists("Op with name ", def.name());
  }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

// static
OpRegistry* OpRegistry::Global() {
  static OpRegistry* global_op_registry = new OpRegistry;
  return global_op_registry;
}

<<<<<<< HEAD
// OpListOpRegistry -----------------------------------------------------------

OpListOpRegistry::OpListOpRegistry(const OpList* op_list) {
  for (const OpDef& op_def : op_list->op()) {
    auto* op_reg_data = new OpRegistrationData();
    op_reg_data->op_def = op_def;
    index_[op_def.name()] = op_reg_data;
  }
}

OpListOpRegistry::~OpListOpRegistry() {
  for (const auto& e : index_) delete e.second;
}

const OpRegistrationData* OpListOpRegistry::LookUp(
    const string& op_type_name) const {
  auto iter = index_.find(op_type_name);
  if (iter == index_.end()) {
    return nullptr;
  }
  return iter->second;
}

Status OpListOpRegistry::LookUp(const string& op_type_name,
                                const OpRegistrationData** op_reg_data) const {
  if ((*op_reg_data = LookUp(op_type_name))) return Status::OK();
  return OpNotFound(op_type_name);
}

// Other registration ---------------------------------------------------------

namespace register_op {
OpDefBuilderReceiver::OpDefBuilderReceiver(
    const OpDefBuilderWrapper<true>& wrapper) {
  OpRegistry::Global()->Register(
      [wrapper](OpRegistrationData* op_reg_data) -> Status {
        return wrapper.builder().Finalize(op_reg_data);
      });
=======
namespace register_op {
OpDefBuilder& RegisterOp(StringPiece name) {
  VLOG(1) << "RegisterOp: " << name;
  OpDefBuilder* b = new OpDefBuilder(name);
  OpRegistry::Global()->Register([b]() -> ::tensorflow::OpDef {
    OpDef op_def;
    TF_QCHECK_OK(b->Finalize(&op_def));
    delete b;
    return op_def;
  });
  return *b;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}
}  // namespace register_op

}  // namespace tensorflow
