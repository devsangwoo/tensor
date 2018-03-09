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
#include "tensorflow/contrib/lite/util.h"

namespace tflite {

TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input) {
  TfLiteIntArray* output = TfLiteIntArrayCreate(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    output->data[i] = input[i];
  }
  return output;
}

bool EqualVectorAndTfLiteIntArray(const TfLiteIntArray* a,
                                  const std::vector<int>& b) {
  if (!a) return false;
  if (a->size != b.size()) return false;
  for (int i = 0; i < a->size; ++i) {
    if (a->data[i] != b[i]) return false;
  }
  return true;
}

}  // namespace tflite
