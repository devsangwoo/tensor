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

#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"

#if !defined(PLATFORM_WINDOWS)
#include <dirent.h>
#endif

#include <limits.h>
=======
#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"

#include <dirent.h>
#include <limits.h>
#include <link.h>
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
<<<<<<< HEAD
#ifdef __APPLE__
#include <IOKit/kext/KextManager.h>
#include <mach-o/dyld.h>
#else
#if !defined(PLATFORM_WINDOWS)
#include <link.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif
#include <sys/stat.h>
#endif
=======
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include <algorithm>
#include <memory>
#include <vector>

<<<<<<< HEAD
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/numbers.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace stream_executor {
namespace cuda {

string DriverVersionToString(DriverVersion version) {
  return absl::StrFormat("%d.%d.%d", std::get<0>(version), std::get<1>(version),
                         std::get<2>(version));
=======
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/inlined_vector.h"
#include "tensorflow/stream_executor/lib/numbers.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringpiece.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace perftools {
namespace gputools {
namespace cuda {

static const char *kDriverVersionPath = "/proc/driver/nvidia/version";

string DriverVersionToString(DriverVersion version) {
  return port::Printf("%d.%d", std::get<0>(version), std::get<1>(version));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

string DriverVersionStatusToString(port::StatusOr<DriverVersion> version) {
  if (!version.ok()) {
    return version.status().ToString();
  }

  return DriverVersionToString(version.ValueOrDie());
}

port::StatusOr<DriverVersion> StringToDriverVersion(const string &value) {
<<<<<<< HEAD
  std::vector<string> pieces = absl::StrSplit(value, '.');
  if (pieces.size() < 2 || pieces.size() > 4) {
    return port::Status(
        port::error::INVALID_ARGUMENT,
        absl::StrFormat(
            "expected %%d.%%d, %%d.%%d.%%d, or %%d.%%d.%%d.%%d form "
            "for driver version; got \"%s\"",
            value.c_str()));
=======
  std::vector<string> pieces = port::Split(value, '.');
  if (pieces.size() != 2) {
    return port::Status{
        port::error::INVALID_ARGUMENT,
        port::Printf("expected %%d.%%d form for driver version; got \"%s\"",
                     value.c_str())};
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  int major;
  int minor;
<<<<<<< HEAD
  int patch = 0;
  if (!port::safe_strto32(pieces[0], &major)) {
    return port::Status(
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("could not parse major version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[0], value));
  }
  if (!port::safe_strto32(pieces[1], &minor)) {
    return port::Status(
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("could not parse minor version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[1].c_str(), value.c_str()));
  }
  if (pieces.size() == 3 && !port::safe_strto32(pieces[2], &patch)) {
    return port::Status(
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("could not parse patch version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[2], value));
  }

  DriverVersion result{major, minor, patch};
=======
  if (!port::safe_strto32(pieces[0], &major)) {
    return port::Status{
        port::error::INVALID_ARGUMENT,
        port::Printf("could not parse major version number \"%s\" as an "
                     "integer from string \"%s\"",
                     pieces[0].c_str(), value.c_str())};
  }
  if (!port::safe_strto32(pieces[1], &minor)) {
    return port::Status{
        port::error::INVALID_ARGUMENT,
        port::Printf("could not parse minor version number \"%s\" as an "
                     "integer from string \"%s\"",
                     pieces[1].c_str(), value.c_str())};
  }

  DriverVersion result{major, minor};
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  VLOG(2) << "version string \"" << value << "\" made value "
          << DriverVersionToString(result);
  return result;
}

<<<<<<< HEAD
}  // namespace cuda
}  // namespace stream_executor

namespace stream_executor {
namespace gpu {

#ifdef __APPLE__
static const CFStringRef kDriverKextIdentifier = CFSTR("com.nvidia.CUDA");
#elif !defined(PLATFORM_WINDOWS)
static const char *kDriverVersionPath = "/proc/driver/nvidia/version";
#endif

// -- class Diagnostician

string Diagnostician::GetDevNodePath(int dev_node_ordinal) {
  return absl::StrCat("/dev/nvidia", dev_node_ordinal);
}

void Diagnostician::LogDiagnosticInformation() {
#ifdef __APPLE__
  CFStringRef kext_ids[1];
  kext_ids[0] = kDriverKextIdentifier;
  CFArrayRef kext_id_query = CFArrayCreate(nullptr, (const void **)kext_ids, 1,
                                           &kCFTypeArrayCallBacks);
  CFDictionaryRef kext_infos =
      KextManagerCopyLoadedKextInfo(kext_id_query, nullptr);
  CFRelease(kext_id_query);

  CFDictionaryRef cuda_driver_info = nullptr;
  if (CFDictionaryGetValueIfPresent(kext_infos, kDriverKextIdentifier,
                                    (const void **)&cuda_driver_info)) {
    bool started = CFBooleanGetValue((CFBooleanRef)CFDictionaryGetValue(
        cuda_driver_info, CFSTR("OSBundleStarted")));
    if (!started) {
      LOG(INFO) << "kernel driver is installed, but does not appear to be "
                   "running on this host "
                << "(" << port::Hostname() << ")";
    }
  } else {
    LOG(INFO) << "kernel driver does not appear to be installed on this host "
              << "(" << port::Hostname() << ")";
  }
  CFRelease(kext_infos);
#elif !defined(PLATFORM_WINDOWS)
=======
// -- class Diagnostician

string Diagnostician::GetDevNodePath(int dev_node_ordinal) {
  return port::StrCat("/dev/nvidia", dev_node_ordinal);
}

void Diagnostician::LogDiagnosticInformation() {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (access(kDriverVersionPath, F_OK) != 0) {
    LOG(INFO) << "kernel driver does not appear to be running on this host "
              << "(" << port::Hostname() << "): "
              << "/proc/driver/nvidia/version does not exist";
    return;
  }
  auto dev0_path = GetDevNodePath(0);
  if (access(dev0_path.c_str(), F_OK) != 0) {
    LOG(INFO) << "no NVIDIA GPU device is present: " << dev0_path
              << " does not exist";
    return;
  }
<<<<<<< HEAD
#endif
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  LOG(INFO) << "retrieving CUDA diagnostic information for host: "
            << port::Hostname();

<<<<<<< HEAD
=======

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  LogDriverVersionInformation();
}

/* static */ void Diagnostician::LogDriverVersionInformation() {
  LOG(INFO) << "hostname: " << port::Hostname();
<<<<<<< HEAD
#ifndef PLATFORM_WINDOWS
=======

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (VLOG_IS_ON(1)) {
    const char *value = getenv("LD_LIBRARY_PATH");
    string library_path = value == nullptr ? "" : value;
    VLOG(1) << "LD_LIBRARY_PATH is: \"" << library_path << "\"";

<<<<<<< HEAD
    std::vector<string> pieces = absl::StrSplit(library_path, ':');
    for (const auto &piece : pieces) {
=======
    std::vector<string> pieces = port::Split(library_path, ':');
    for (auto piece : pieces) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      if (piece.empty()) {
        continue;
      }
      DIR *dir = opendir(piece.c_str());
      if (dir == nullptr) {
        VLOG(1) << "could not open \"" << piece << "\"";
        continue;
      }
      while (dirent *entity = readdir(dir)) {
        VLOG(1) << piece << " :: " << entity->d_name;
      }
      closedir(dir);
    }
  }
<<<<<<< HEAD
  port::StatusOr<DriverVersion> dso_version = FindDsoVersion();
  LOG(INFO) << "libcuda reported version is: "
            << cuda::DriverVersionStatusToString(dso_version);

  port::StatusOr<DriverVersion> kernel_version = FindKernelDriverVersion();
  LOG(INFO) << "kernel reported version is: "
            << cuda::DriverVersionStatusToString(kernel_version);
#endif

  // OS X kernel driver does not report version accurately
#if !defined(__APPLE__) && !defined(PLATFORM_WINDOWS)
  if (kernel_version.ok() && dso_version.ok()) {
    WarnOnDsoKernelMismatch(dso_version, kernel_version);
  }
#endif
=======

  port::StatusOr<DriverVersion> dso_version = FindDsoVersion();
  LOG(INFO) << "libcuda reported version is: "
            << DriverVersionStatusToString(dso_version);

  port::StatusOr<DriverVersion> kernel_version = FindKernelDriverVersion();
  LOG(INFO) << "kernel reported version is: "
            << DriverVersionStatusToString(kernel_version);
  if (kernel_version.ok() && dso_version.ok()) {
    WarnOnDsoKernelMismatch(dso_version, kernel_version);
  }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

// Iterates through loaded DSOs with DlIteratePhdrCallback to find the
// driver-interfacing DSO version number. Returns it as a string.
port::StatusOr<DriverVersion> Diagnostician::FindDsoVersion() {
<<<<<<< HEAD
  port::StatusOr<DriverVersion> result(port::Status(
      port::error::NOT_FOUND,
      "was unable to find libcuda.so DSO loaded into this program"));

#if defined(__APPLE__)
  // OSX CUDA libraries have names like: libcuda_310.41.15_mercury.dylib
  const string prefix("libcuda_");
  const string suffix("_mercury.dylib");
  for (uint32_t image_index = 0; image_index < _dyld_image_count();
       ++image_index) {
    const string path(_dyld_get_image_name(image_index));
    const size_t suffix_pos = path.rfind(suffix);
    const size_t prefix_pos = path.rfind(prefix, suffix_pos);
    if (prefix_pos == string::npos || suffix_pos == string::npos) {
      // no match
      continue;
    }
    const size_t start = prefix_pos + prefix.size();
    if (start >= suffix_pos) {
      // version not included
      continue;
    }
    const size_t length = suffix_pos - start;
    const string version = path.substr(start, length);
    result = cuda::StringToDriverVersion(version);
  }
#else
#if !defined(PLATFORM_WINDOWS) && !defined(ANDROID_TEGRA)
=======
  port::StatusOr<DriverVersion> result{port::Status{
      port::error::NOT_FOUND,
      "was unable to find libcuda.so DSO loaded into this program"}};

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  // Callback used when iterating through DSOs. Looks for the driver-interfacing
  // DSO and yields its version number into the callback data, when found.
  auto iterate_phdr =
      [](struct dl_phdr_info *info, size_t size, void *data) -> int {
<<<<<<< HEAD
    if (strstr(info->dlpi_name, "libcuda.so.1")) {
=======
    if (strstr(info->dlpi_name, "libcuda.so")) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      VLOG(1) << "found DLL info with name: " << info->dlpi_name;
      char resolved_path[PATH_MAX] = {0};
      if (realpath(info->dlpi_name, resolved_path) == nullptr) {
        return 0;
      }
      VLOG(1) << "found DLL info with resolved path: " << resolved_path;
      const char *slash = rindex(resolved_path, '/');
      if (slash == nullptr) {
        return 0;
      }
      const char *so_suffix = ".so.";
      const char *dot = strstr(slash, so_suffix);
      if (dot == nullptr) {
        return 0;
      }
      string dso_version = dot + strlen(so_suffix);
      // TODO(b/22689637): Eliminate the explicit namespace if possible.
<<<<<<< HEAD
      auto stripped_dso_version = absl::StripSuffix(dso_version, ".ld64");
      auto result = static_cast<port::StatusOr<DriverVersion> *>(data);
      *result = cuda::StringToDriverVersion(string(stripped_dso_version));
=======
      auto stripped_dso_version = port::StripSuffixString(dso_version, ".ld64");
      auto result = static_cast<port::StatusOr<DriverVersion> *>(data);
      *result = StringToDriverVersion(stripped_dso_version);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      return 1;
    }
    return 0;
  };

  dl_iterate_phdr(iterate_phdr, &result);
<<<<<<< HEAD
#endif
#endif
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  return result;
}

port::StatusOr<DriverVersion> Diagnostician::FindKernelModuleVersion(
    const string &driver_version_file_contents) {
  static const char *kDriverFilePrelude = "Kernel Module  ";
  size_t offset = driver_version_file_contents.find(kDriverFilePrelude);
  if (offset == string::npos) {
<<<<<<< HEAD
    return port::Status(
        port::error::NOT_FOUND,
        absl::StrCat("could not find kernel module information in "
                     "driver version file contents: \"",
                     driver_version_file_contents, "\""));
=======
    return port::Status{
        port::error::NOT_FOUND,
        port::StrCat("could not find kernel module information in "
                     "driver version file contents: \"",
                     driver_version_file_contents, "\"")};
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  string version_and_rest = driver_version_file_contents.substr(
      offset + strlen(kDriverFilePrelude), string::npos);
  size_t space_index = version_and_rest.find(" ");
  auto kernel_version = version_and_rest.substr(0, space_index);
  // TODO(b/22689637): Eliminate the explicit namespace if possible.
<<<<<<< HEAD
  auto stripped_kernel_version = absl::StripSuffix(kernel_version, ".ld64");
  return cuda::StringToDriverVersion(string(stripped_kernel_version));
=======
  auto stripped_kernel_version =
      port::StripSuffixString(kernel_version, ".ld64");
  return StringToDriverVersion(stripped_kernel_version);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

void Diagnostician::WarnOnDsoKernelMismatch(
    port::StatusOr<DriverVersion> dso_version,
    port::StatusOr<DriverVersion> kernel_version) {
  if (kernel_version.ok() && dso_version.ok() &&
      dso_version.ValueOrDie() == kernel_version.ValueOrDie()) {
    LOG(INFO) << "kernel version seems to match DSO: "
<<<<<<< HEAD
              << cuda::DriverVersionToString(kernel_version.ValueOrDie());
  } else {
    LOG(ERROR) << "kernel version "
               << cuda::DriverVersionStatusToString(kernel_version)
               << " does not match DSO version "
               << cuda::DriverVersionStatusToString(dso_version)
=======
              << DriverVersionToString(kernel_version.ValueOrDie());
  } else {
    LOG(ERROR) << "kernel version "
               << DriverVersionStatusToString(kernel_version)
               << " does not match DSO version "
               << DriverVersionStatusToString(dso_version)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
               << " -- cannot find working devices in this configuration";
  }
}


port::StatusOr<DriverVersion> Diagnostician::FindKernelDriverVersion() {
<<<<<<< HEAD
#if defined(__APPLE__)
  CFStringRef kext_ids[1];
  kext_ids[0] = kDriverKextIdentifier;
  CFArrayRef kext_id_query = CFArrayCreate(nullptr, (const void **)kext_ids, 1,
                                           &kCFTypeArrayCallBacks);
  CFDictionaryRef kext_infos =
      KextManagerCopyLoadedKextInfo(kext_id_query, nullptr);
  CFRelease(kext_id_query);

  CFDictionaryRef cuda_driver_info = nullptr;
  if (CFDictionaryGetValueIfPresent(kext_infos, kDriverKextIdentifier,
                                    (const void **)&cuda_driver_info)) {
    // NOTE: OSX CUDA driver does not currently store the same driver version
    // in kCFBundleVersionKey as is returned by cuDriverGetVersion
    CFRelease(kext_infos);
    const CFStringRef str = (CFStringRef)CFDictionaryGetValue(
        cuda_driver_info, kCFBundleVersionKey);
    const char *version = CFStringGetCStringPtr(str, kCFStringEncodingUTF8);

    // version can be NULL in which case treat it as empty string
    // see
    // https://developer.apple.com/library/mac/documentation/CoreFoundation/Conceptual/CFStrings/Articles/AccessingContents.html#//apple_ref/doc/uid/20001184-100980-TPXREF112
    if (version == NULL) {
      return cuda::StringToDriverVersion("");
    }
    return cuda::StringToDriverVersion(version);
  }
  CFRelease(kext_infos);
  auto status = port::Status(
      port::error::INTERNAL,
      absl::StrCat(
          "failed to read driver bundle version: ",
          CFStringGetCStringPtr(kDriverKextIdentifier, kCFStringEncodingUTF8)));
  return status;
#elif defined(PLATFORM_WINDOWS)
  auto status =
      port::Status(port::error::UNIMPLEMENTED,
                   "kernel reported driver version not implemented on Windows");
  return status;
#else
  FILE *driver_version_file = fopen(kDriverVersionPath, "r");
  if (driver_version_file == nullptr) {
    return port::Status(
        port::error::PERMISSION_DENIED,
        absl::StrCat("could not open driver version path for reading: ",
                     kDriverVersionPath));
  }

  static const int kContentsSize = 1024;
  absl::InlinedVector<char, 4> contents(kContentsSize);
=======
  FILE *driver_version_file = fopen(kDriverVersionPath, "r");
  if (driver_version_file == nullptr) {
    return port::Status{
        port::error::PERMISSION_DENIED,
        port::StrCat("could not open driver version path for reading: ",
                     kDriverVersionPath)};
  }

  static const int kContentsSize = 1024;
  port::InlinedVector<char, 4> contents(kContentsSize);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  size_t retcode =
      fread(contents.begin(), 1, kContentsSize - 2, driver_version_file);
  if (retcode < kContentsSize - 1) {
    contents[retcode] = '\0';
  }
  contents[kContentsSize - 1] = '\0';

  if (retcode != 0) {
<<<<<<< HEAD
    VLOG(1) << "driver version file contents: \"\"\"" << contents.begin()
            << "\"\"\"";
    fclose(driver_version_file);
    return FindKernelModuleVersion(contents.begin());
  }

  auto status = port::Status(
      port::error::INTERNAL,
      absl::StrCat(
          "failed to read driver version file contents: ", kDriverVersionPath,
          "; ferror: ", ferror(driver_version_file)));
  fclose(driver_version_file);
  return status;
#endif
}

}  // namespace gpu
}  // namespace stream_executor
=======
    LOG(INFO) << "driver version file contents: \"\"\"" << contents.begin()
              << "\"\"\"";
    fclose(driver_version_file);
    return FindKernelModuleVersion(string{contents.begin()});
  }

  auto status =
      port::Status{port::error::INTERNAL,
                   port::StrCat("failed to read driver version file contents: ",
                                kDriverVersionPath, "; ferror: ",
                                ferror(driver_version_file))};
  fclose(driver_version_file);
  return status;
}


}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
