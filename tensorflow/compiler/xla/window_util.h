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

#ifndef TENSORFLOW_COMPILER_XLA_WINDOW_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_WINDOW_UTIL_H_

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace window_util {

string ToString(const WindowDimension& dim);
string ToString(const Window& window);

// The below functions return true if the given field is set to have a
// non-trivial effect, e.g. having a stride means that the stride of some
// dimension is not one. Whether the proto field is populated is not a
// consideration.

bool HasStride(const Window& window);
bool HasPadding(const Window& window);
bool HasEvenPadding(const Window& window);
bool HasNegativePadding(const Window& window);

bool HasBaseDilation(const Window& window);
bool HasWindowDilation(const Window& window);
bool HasDilation(const Window& window);

// Returns the new bound after dilation.
//
// If a window with the given bound in some dimension is dilated with the given
// dilation factor in that dimension, then the value returned is the bound for
// the array in that dimension after dilation.
//
// For a 1D array with 3 entries 1, 2, 3, a dilation factor of 2 yields a new
// window with values 1, x, 2, x, 3, where x indicates holes left by the
// dilation. So DilatedBound(3, 2) == 5.
int64 DilatedBound(int64 bound, int64 dilation);

// Returns the number of valid positions of a window with the given size and
// stride within an array with the given bound. This is the bound of an output
// array with one element per valid position of the window.
//
// For example, for arguments of (bound=5, window_size=2, stride=2), the
// returned value is 2. There are valid positions at offset 0 and offset 2,
// while offset 4 is not valid since the window's last entry would be at 5,
// which is beyond the bound of 5.
int64 StridedBound(int64 bound, int64 window_size, int64 stride);

}  // namespace window_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_WINDOW_UTIL_H_
