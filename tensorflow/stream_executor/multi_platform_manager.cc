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

#include "tensorflow/stream_executor/multi_platform_manager.h"

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"

namespace stream_executor {
namespace {

class MultiPlatformManagerImpl {
 public:
  port::Status RegisterPlatform(std::unique_ptr<Platform> platform)
      LOCKS_EXCLUDED(mu_);

  port::StatusOr<Platform*> PlatformWithName(absl::string_view target)
      LOCKS_EXCLUDED(mu_);

  port::StatusOr<Platform*> PlatformWithId(const Platform::Id& id)
      LOCKS_EXCLUDED(mu_);

  port::StatusOr<Platform*> InitializePlatformWithName(
      absl::string_view target, const std::map<string, string>& options)
      LOCKS_EXCLUDED(mu_);
  port::StatusOr<Platform*> InitializePlatformWithId(
      const Platform::Id& id, const std::map<string, string>& options)
      LOCKS_EXCLUDED(mu_);

  port::StatusOr<std::vector<Platform*>> PlatformsWithFilter(
      const std::function<bool(const Platform*)>& filter) LOCKS_EXCLUDED(mu_);

  using Listener = MultiPlatformManager::Listener;
  port::Status RegisterListener(std::unique_ptr<Listener> listener)
      LOCKS_EXCLUDED(mu_);

 private:
  // Looks up the platform object with the given name.  Assumes the Platforms
  // mutex is held.
  port::StatusOr<Platform*> LookupByNameLocked(absl::string_view target)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Looks up the platform object with the given id.  Assumes the Platforms
  // mutex is held.
  port::StatusOr<Platform*> LookupByIdLocked(const Platform::Id& id)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Mutex mu_;
  std::vector<std::unique_ptr<Listener>> listeners_ GUARDED_BY(mu_);
  absl::flat_hash_map<Platform::Id, Platform*> id_map_ GUARDED_BY(mu_);
  absl::flat_hash_map<string, Platform*> name_map_ GUARDED_BY(mu_);
};

port::Status MultiPlatformManagerImpl::RegisterPlatform(
    std::unique_ptr<Platform> platform) {
  CHECK(platform != nullptr);
  string key = absl::AsciiStrToLower(platform->Name());
  absl::MutexLock lock(&mu_);
  if (name_map_.find(key) != name_map_.end()) {
=======
#include "tensorflow/stream_executor/multi_platform_manager.h"

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {

/* static */ mutex MultiPlatformManager::platforms_mutex_(LINKER_INITIALIZED);

/* static */ port::Status MultiPlatformManager::RegisterPlatform(
    std::unique_ptr<Platform> platform) {
  CHECK(platform != nullptr);
  string key = port::Lowercase(platform->Name());
  mutex_lock lock(platforms_mutex_);
  if (GetPlatformMap()->find(key) != GetPlatformMap()->end()) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    return port::Status(port::error::INTERNAL,
                        "platform is already registered with name: \"" +
                            platform->Name() + "\"");
  }
<<<<<<< HEAD
  Platform* platform_ptr = platform.get();
  CHECK(id_map_.emplace(platform->id(), platform_ptr).second);
=======
  GetPlatformByIdMap()->insert(std::make_pair(platform->id(), platform.get()));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // Release ownership/uniqueness to prevent destruction on program exit.
  // This avoids Platforms "cleaning up" on program exit, because otherwise,
  // there are _very_ tricky races between StreamExecutor and underlying
  // platforms (CUDA, OpenCL) during exit. Since these are fixed-size and 1x per
  // program, these are deemed acceptable.
<<<<<<< HEAD
  name_map_[key] = platform.release();
  for (const auto& listener : listeners_) {
    listener->PlatformRegistered(platform_ptr);
  }
  return port::Status::OK();
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::PlatformWithName(
    absl::string_view target) {
  absl::MutexLock lock(&mu_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByNameLocked(target));
  if (!platform->Initialized()) {
    SE_RETURN_IF_ERROR(platform->Initialize({}));
  }

  return platform;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::PlatformWithId(
    const Platform::Id& id) {
  absl::MutexLock lock(&mu_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByIdLocked(id));
  if (!platform->Initialized()) {
    SE_RETURN_IF_ERROR(platform->Initialize({}));
  }

  return platform;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::InitializePlatformWithName(
    absl::string_view target, const std::map<string, string>& options) {
  absl::MutexLock lock(&mu_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByNameLocked(target));
  if (platform->Initialized()) {
    return port::Status(
        port::error::FAILED_PRECONDITION,
        absl::StrCat("platform \"", target, "\" is already initialized"));
  }

  SE_RETURN_IF_ERROR(platform->Initialize(options));

  return platform;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::InitializePlatformWithId(
    const Platform::Id& id, const std::map<string, string>& options) {
  absl::MutexLock lock(&mu_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByIdLocked(id));
  if (platform->Initialized()) {
    return port::Status(
        port::error::FAILED_PRECONDITION,
        absl::StrFormat("platform with id %p is already initialized", id));
  }

  SE_RETURN_IF_ERROR(platform->Initialize(options));

  return platform;
}

port::Status MultiPlatformManagerImpl::RegisterListener(
    std::unique_ptr<Listener> listener) {
  absl::MutexLock lock(&mu_);
  CHECK(id_map_.empty());
  CHECK(name_map_.empty());
  listeners_.push_back(std::move(listener));
  return port::Status::OK();
}

port::StatusOr<std::vector<Platform*>>
MultiPlatformManagerImpl::PlatformsWithFilter(
    const std::function<bool(const Platform*)>& filter) {
  absl::MutexLock lock(&mu_);
  CHECK_EQ(id_map_.size(), name_map_.size());
  std::vector<Platform*> platforms;
  platforms.reserve(id_map_.size());
  for (const auto& entry : id_map_) {
    Platform* platform = entry.second;
    if (filter(platform)) {
      if (!platform->Initialized()) {
        SE_RETURN_IF_ERROR(platform->Initialize({}));
      }
      platforms.push_back(platform);
    }
  }
  return platforms;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::LookupByNameLocked(
    absl::string_view target) {
  auto it = name_map_.find(absl::AsciiStrToLower(target));
  if (it == name_map_.end()) {
    return port::Status(
        port::error::NOT_FOUND,
        absl::StrCat("Could not find registered platform with name: \"", target,
                     "\""));
  }
  return it->second;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::LookupByIdLocked(
    const Platform::Id& id) {
  auto it = id_map_.find(id);
  if (it == id_map_.end()) {
    return port::Status(
        port::error::NOT_FOUND,
        absl::StrFormat("could not find registered platform with id: %p", id));
  }
  return it->second;
}

MultiPlatformManagerImpl& Impl() {
  static MultiPlatformManagerImpl* impl = new MultiPlatformManagerImpl;
  return *impl;
}

}  // namespace

/*static*/ port::Status MultiPlatformManager::RegisterPlatform(
    std::unique_ptr<Platform> platform) {
  return Impl().RegisterPlatform(std::move(platform));
}

/*static*/ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithName(
    absl::string_view target) {
  return Impl().PlatformWithName(target);
}

/*static*/ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithId(
    const Platform::Id& id) {
  return Impl().PlatformWithId(id);
}

/*static*/ port::StatusOr<Platform*>
MultiPlatformManager::InitializePlatformWithName(
    absl::string_view target, const std::map<string, string>& options) {
  return Impl().InitializePlatformWithName(target, options);
}

/*static*/ port::StatusOr<Platform*>
MultiPlatformManager::InitializePlatformWithId(
    const Platform::Id& id, const std::map<string, string>& options) {
  return Impl().InitializePlatformWithId(id, options);
}

/*static*/ port::Status MultiPlatformManager::RegisterListener(
    std::unique_ptr<Listener> listener) {
  return Impl().RegisterListener(std::move(listener));
}

/*static*/ port::StatusOr<std::vector<Platform*>>
MultiPlatformManager::PlatformsWithFilter(
    const std::function<bool(const Platform*)>& filter) {
  return Impl().PlatformsWithFilter(filter);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(
    multi_platform_manager,
    {
        // Nothing -- this is just a module initializer
        // definition to reference for sequencing
        // purposes from Platform subclasses that register
        // themselves with the MultiPlatformManager.
    });

REGISTER_MODULE_INITIALIZER(
    multi_platform_manager_listener,
    {
        // Nothing -- this is just a module initializer definition to reference
        // for sequencing registration of listeners with the
        // MultiPlatformManager.
    });

// Listener registration should happen before platform registration.
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     multi_platform_manager);
=======
  (*GetPlatformMap())[key] = platform.release();
  return port::Status::OK();
}

/* static */ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithName(
    const string& target) {
  mutex_lock lock(platforms_mutex_);
  auto it = GetPlatformMap()->find(port::Lowercase(target));

  if (it == GetPlatformMap()->end()) {
    return port::Status(
        port::error::NOT_FOUND,
        "could not find registered platform with name: \"" + target + "\"");
  }

  return it->second;
}

/* static */ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithId(
    const Platform::Id& id) {
  mutex_lock lock(platforms_mutex_);
  auto it = GetPlatformByIdMap()->find(id);
  if (it == GetPlatformByIdMap()->end()) {
    return port::Status(
        port::error::NOT_FOUND,
        port::Printf("could not find registered platform with id: 0x%p", id));
  }

  return it->second;
}

/* static */ void MultiPlatformManager::ClearPlatformRegistry() {
  mutex_lock lock(platforms_mutex_);
  GetPlatformMap()->clear();
  GetPlatformByIdMap()->clear();
}

}  // namespace gputools
}  // namespace perftools
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
