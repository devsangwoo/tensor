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

#include "tensorflow/core/platform/tracing.h"

#include <array>
#include <atomic>

#include "tensorflow/core/platform/hash.h"

namespace tensorflow {
namespace tracing {
namespace {
std::atomic<uint64> unique_arg{1};
}  // namespace

const char* GetEventCategoryName(EventCategory category) {
=======
#include "tensorflow/core/platform/tracing.h"

#include <atomic>
#include <map>
#include <string>
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

StepStatsCollector::StepStatsCollector(StepStats* ss) : step_stats_(ss) {}

void StepStatsCollector::Save(const string& device, NodeExecStats* nt) {
  VLOG(1) << "Save dev " << device << " nt " << nt;
  {
    mutex_lock l(mu_);
    DeviceStepStats* dss = nullptr;
    // Slow linear scan, but it should only be called
    // by a Worker in a context with < ~10 devices.
    // TODO(tucker): consider adding a std::unordered_map.
    for (auto& ds : *step_stats_->mutable_dev_stats()) {
      if (ds.device() == device) {
        dss = &ds;
        break;
      }
    }
    if (dss == nullptr) {
      dss = step_stats_->add_dev_stats();
      dss->set_device(device);
    }
    nt->Swap(dss->add_node_stats());
  }
  delete nt;
}

void StepStatsCollector::Swap(StepStats* ss) {
  mutex_lock l(mu_);
  CHECK(step_stats_);
  ss->Swap(step_stats_);
}

namespace port {

int32 Tracing::category_id_[kEventCategoryMax];
uint64 Tracing::event_mask_ = 0;
std::map<string, int32>* Tracing::name_map_ = new std::map<string, int32>;

// This needs to be kept in sync with the EventCategory enumeration.
const char* Tracing::EventCategoryString(EventCategory category) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  switch (category) {
    case EventCategory::kScheduleClosure:
      return "ScheduleClosure";
    case EventCategory::kRunClosure:
      return "RunClosure";
    case EventCategory::kCompute:
      return "Compute";
<<<<<<< HEAD
    default:
      return "Unknown";
  }
}

std::array<const EventCollector*, GetNumEventCategories()>
    EventCollector::instances_;

void SetEventCollector(EventCategory category,
                       const EventCollector* collector) {
  EventCollector::instances_[static_cast<unsigned>(category)] = collector;
}

uint64 GetUniqueArg() {
  return unique_arg.fetch_add(1, std::memory_order_relaxed);
}

uint64 GetArgForName(StringPiece name) {
  return Hash64(name.data(), name.size());
}

}  // namespace tracing
=======
    case EventCategory::kEventCategoryMax:
      return "EventCategoryMax";
  }
  return "Unknown";
}

// This function allows the user to specify arbitrary subsets of the
// supported Threadscape events and activities.
bool Tracing::ParseEventMask(const char* flagname, const string& value) {
  VLOG(1) << flagname << " set to " << value;
  int64 new_mask = 0;
  std::vector<string> events =
      str_util::Split(value, ',', str_util::SkipEmpty());
  for (string name : events) {
    bool clear = false;
    int64 mask = 0;
    if (name[0] == '!') {
      // invert the sense of the flag
      clear = true;
      name = name.substr(1);
    }
    if (name == "ALL") {
      mask = ~0;
    } else {
      auto it = name_map_->find(name);
      int32 id;
      if (it == name_map_->end()) {
        id = -1;
      } else {
        id = it->second;
      }
      if (id < 0) {
        LOG(ERROR) << "Can't parse event mask name " << name;
        return false;
      }
      mask = 1 << id;
    }
    if (clear) {
      new_mask &= ~mask;
    } else {
      new_mask |= mask;
    }
  }
  // parsing was successful; set the permanent event mask
  event_mask_ = new_mask;
  return true;
}

static std::atomic<Tracing::Engine*> tracing_engine;

void Tracing::RegisterEngine(Engine* e) {
  tracing_engine.store(e, std::memory_order_release);
}

static Tracing::Engine* engine() {
  return tracing_engine.load(std::memory_order_acquire);
}

Tracing::Engine::~Engine() {}
Tracing::Engine::Annotation::~Annotation() {}
Tracing::Engine::Tracer::~Tracer() {}

Tracing::ScopedAnnotation::ScopedAnnotation(StringPiece name) {
  auto e = engine();
  if (e) {
    annotation_.reset(e->PushAnnotation(name));
  }
}

Tracing::TraceMe::TraceMe(StringPiece name) {
  auto e = engine();
  if (e) {
    tracer_.reset(e->StartTracing(name));
  }
}

}  // namespace port
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
