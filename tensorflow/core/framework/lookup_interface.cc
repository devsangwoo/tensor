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

#include "tensorflow/core/framework/lookup_interface.h"

#include "tensorflow/core/framework/tensor_shape.h"
=======
#include "tensorflow/core/framework/lookup_interface.h"

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lookup {

<<<<<<< HEAD
Status LookupInterface::CheckKeyShape(const TensorShape& shape) {
  if (!TensorShapeUtils::EndsWith(shape, key_shape())) {
    return errors::InvalidArgument("Input key shape ", shape.DebugString(),
                                   " must end with the table's key shape ",
                                   key_shape().DebugString());
  }
  return Status::OK();
}

Status LookupInterface::CheckKeyAndValueTypes(const Tensor& keys,
                                              const Tensor& values) {
  if (keys.dtype() != key_dtype()) {
    return errors::InvalidArgument("Key must be type ", key_dtype(),
                                   " but got ", keys.dtype());
  }
  if (values.dtype() != value_dtype()) {
    return errors::InvalidArgument("Value must be type ", value_dtype(),
                                   " but got ", values.dtype());
  }
  return Status::OK();
}

Status LookupInterface::CheckKeyAndValueTensorsHelper(const Tensor& keys,
                                                      const Tensor& values) {
  TF_RETURN_IF_ERROR(CheckKeyAndValueTypes(keys, values));
  TF_RETURN_IF_ERROR(CheckKeyShape(keys.shape()));

  TensorShape expected_value_shape = keys.shape();
  for (int i = 0; i < key_shape().dims(); ++i) {
    expected_value_shape.RemoveDim(expected_value_shape.dims() - 1);
  }
  expected_value_shape.AppendShape(value_shape());
  if (values.shape() != expected_value_shape) {
    return errors::InvalidArgument(
        "Expected shape ", expected_value_shape.DebugString(),
        " for value, got ", values.shape().DebugString());
=======
Status LookupInterface::CheckKeyAndValueTensors(const Tensor& key,
                                                const Tensor& value) {
  if (key.dtype() != key_dtype()) {
    return errors::InvalidArgument("Key must be type ", key_dtype(),
                                   " but got ", key.dtype());
  }
  if (value.dtype() != value_dtype()) {
    return errors::InvalidArgument("Value must be type ", value_dtype(),
                                   " but got ", value.dtype());
  }
  if (key.NumElements() != value.NumElements()) {
    return errors::InvalidArgument("Number of elements of key(",
                                   key.NumElements(), ") and value(",
                                   value.NumElements(), ") are different.");
  }
  if (!key.shape().IsSameSize(value.shape())) {
    return errors::InvalidArgument("key and value have different shapes.");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
  return Status::OK();
}

<<<<<<< HEAD
Status LookupInterface::CheckKeyAndValueTensorsForInsert(const Tensor& keys,
                                                         const Tensor& values) {
  return CheckKeyAndValueTensorsHelper(keys, values);
}

Status LookupInterface::CheckKeyAndValueTensorsForImport(const Tensor& keys,
                                                         const Tensor& values) {
  return CheckKeyAndValueTensorsHelper(keys, values);
}

Status LookupInterface::CheckKeyTensorForRemove(const Tensor& keys) {
  if (keys.dtype() != key_dtype()) {
    return errors::InvalidArgument("Key must be type ", key_dtype(),
                                   " but got ", keys.dtype());
  }
  return CheckKeyShape(keys.shape());
}

Status LookupInterface::CheckFindArguments(const Tensor& key,
                                           const Tensor& default_value) {
  TF_RETURN_IF_ERROR(CheckKeyAndValueTypes(key, default_value));
  TF_RETURN_IF_ERROR(CheckKeyShape(key.shape()));
  if (default_value.shape() != value_shape()) {
    return errors::InvalidArgument(
        "Expected shape ", value_shape().DebugString(),
        " for default value, got ", default_value.shape().DebugString());
=======
Status LookupInterface::CheckFindArguments(const Tensor& key,
                                           const Tensor& value,
                                           const Tensor& default_value) {
  TF_RETURN_IF_ERROR(CheckKeyAndValueTensors(key, value));

  if (default_value.dtype() != value_dtype()) {
    return errors::InvalidArgument("Default value must be type ", value_dtype(),
                                   " but got ", default_value.dtype());
  }
  if (!TensorShapeUtils::IsScalar(default_value.shape())) {
    return errors::InvalidArgument("Default values must be scalar.");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
  return Status::OK();
}

}  // namespace lookup
}  // namespace tensorflow
