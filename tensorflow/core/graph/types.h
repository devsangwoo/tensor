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

#ifndef TENSORFLOW_CORE_GRAPH_TYPES_H_
#define TENSORFLOW_CORE_GRAPH_TYPES_H_

#include "tensorflow/core/lib/gtl/int_type.h"
#include "tensorflow/core/platform/types.h"
=======
#ifndef TENSORFLOW_GRAPH_TYPES_H_
#define TENSORFLOW_GRAPH_TYPES_H_

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/lib/gtl/int_type.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

// We model running time in microseconds.
TF_LIB_GTL_DEFINE_INT_TYPE(Microseconds, int64);

<<<<<<< HEAD
// We can also model running time in nanoseconds for more accuracy.
TF_LIB_GTL_DEFINE_INT_TYPE(Nanoseconds, int64);

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// We model size in bytes.
TF_LIB_GTL_DEFINE_INT_TYPE(Bytes, int64);

}  // namespace tensorflow

<<<<<<< HEAD
#endif  // TENSORFLOW_CORE_GRAPH_TYPES_H_
=======
#endif  // TENSORFLOW_GRAPH_TYPES_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
