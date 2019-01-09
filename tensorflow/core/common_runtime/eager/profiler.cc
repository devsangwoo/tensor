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

#include "tensorflow/core/common_runtime/eager/profiler.h"
#include "tensorflow/cc/profiler/profiler.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

/*static*/ std::unique_ptr<EagerProfiler> EagerProfiler::Create(
    EagerContext* const context) {
  return absl::WrapUnique(new EagerProfiler(context));
}

void EagerProfiler::BeforeClearRunMetadata() {
  mutex_lock l(mutex_);
  run_metadata_.MergeFrom(*context_->RunMetadataProto());
}

Status EagerProfiler::Status() {
  mutex_lock l(mutex_);
  return status_;
}

Status EagerProfiler::SerializeToString(string* content) {
  {
    mutex_lock l(mutex_);
    if (!status_.ok()) return status_;
  }
  RunMetadata metadata;
  GetMergetRunMetadata(&metadata);

  // TODO(fishx): update tfprof to use a lighter representation instead of
  // GraphDef.
  GraphDef graph;
  std::unique_ptr<tfprof::Profiler> tfprof(new tfprof::Profiler(graph));
  tfprof->AddStep(0, metadata);
  return tfprof->SerializeToString(content);
}

EagerProfiler::EagerProfiler(EagerContext* const context) : context_(context) {
  LOG(INFO) << "Eager Profiler started.";

  status_ = context_->RegisterRunMetadataListener(this);
  if (!status_.ok()) {
    LOG(INFO) << "Eager Profiler failed to start. Another profiler is running.";
    return;
  }
}

EagerProfiler::~EagerProfiler() {
  context_->ClearRunMetadataListener();
  LOG(INFO) << "Eager Profiler ended with status:" << status_;
}

void EagerProfiler::GetMergetRunMetadata(RunMetadata* metadata) {
  mutex_lock ml(*context_->MetadataMu());
  mutex_lock l(mutex_);
  *metadata = run_metadata_;
  metadata->MergeFrom(*context_->RunMetadataProto());
}

}  // namespace tensorflow
