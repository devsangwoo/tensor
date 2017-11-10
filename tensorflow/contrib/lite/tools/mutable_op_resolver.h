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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOOLS_MUTABLE_OP_RESOLVER_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOOLS_MUTABLE_OP_RESOLVER_H_

#include <unordered_map>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {

// An OpResolver that is mutable, also used as the op in gen_op_registration.
// A typical usage:
//   MutableOpResolver resolver;
//   resolver.AddBuiltin(BuiltinOperator_ADD, Register_ADD());
//   resolver.AddCustom("CustomOp", Register_CUSTOM_OP());
//   InterpreterBuilder(model, resolver)(&interpreter);
class MutableOpResolver : public OpResolver {
 public:
  MutableOpResolver() {}
  TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const override;
  TfLiteRegistration* FindOp(const char* op) const override;
  void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration);
  void AddCustom(const char* name, TfLiteRegistration* registration);

 private:
  std::unordered_map<tflite::BuiltinOperator, TfLiteRegistration*> builtins_;
  std::unordered_map<std::string, TfLiteRegistration*> custom_ops_;
};

}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOOLS_MUTABLE_OP_RESOLVER_H_
