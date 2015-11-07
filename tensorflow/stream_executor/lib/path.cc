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

#include "tensorflow/stream_executor/lib/path.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace stream_executor {
namespace port {
namespace internal {

static bool IsAbsolutePath(absl::string_view path) {
=======
#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/strcat.h"

using ::perftools::gputools::port::StringPiece;
using ::perftools::gputools::port::StrAppend;

namespace perftools {
namespace gputools {
namespace port {
namespace internal {

static bool IsAbsolutePath(port::StringPiece path) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return !path.empty() && path[0] == '/';
}

// For an array of paths of length count, append them all together,
// ensuring that the proper path separators are inserted between them.
<<<<<<< HEAD
string JoinPathImpl(std::initializer_list<absl::string_view> paths) {
  string result;

  for (absl::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = string(path);
=======
string JoinPathImpl(std::initializer_list<port::StringPiece> paths) {
  string result;

  for (port::StringPiece path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = path.ToString();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      continue;
    }

    if (result[result.size() - 1] == '/') {
      if (IsAbsolutePath(path)) {
<<<<<<< HEAD
        absl::StrAppend(&result, path.substr(1));
      } else {
        absl::StrAppend(&result, path);
      }
    } else {
      if (IsAbsolutePath(path)) {
        absl::StrAppend(&result, path);
      } else {
        absl::StrAppend(&result, "/", path);
=======
        StrAppend(&result, path.substr(1));
      } else {
        StrAppend(&result, path);
      }
    } else {
      if (IsAbsolutePath(path)) {
        StrAppend(&result, path);
      } else {
        StrAppend(&result, "/", path);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      }
    }
  }

  return result;
}

}  // namespace internal
}  // namespace port
<<<<<<< HEAD
}  // namespace stream_executor
=======
}  // namespace gputools
}  // namespace perftools
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
