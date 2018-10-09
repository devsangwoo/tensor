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
#include "tensorflow/contrib/lite/toco/tflite/export.h"

#include "flatbuffers/flexbuffers.h"
#include "absl/strings/str_join.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/toco/tflite/operator.h"
#include "tensorflow/contrib/lite/toco/tflite/types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/contrib/lite/tools/optimize/quantize_weights.h"
#include "tensorflow/contrib/lite/version.h"

namespace toco {

namespace tflite {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;
using ::tflite::Buffer;
using ::tflite::BuiltinOperator;
using ::tflite::BuiltinOperator_CUSTOM;
using ::tflite::BuiltinOperator_MAX;
using ::tflite::BuiltinOperator_MIN;
using ::tflite::CreateBuffer;
using ::tflite::CreateModel;
using ::tflite::CreateOperator;
using ::tflite::CreateTensor;
using ::tflite::Operator;
using ::tflite::OperatorCode;
using ::tflite::SubGraph;
using ::tflite::Tensor;

namespace {

// Check if a TensorFlow Op is a control flow op by its name.
bool IsControlFlowOp(const string& tensorflow_op) {
  // Technically this is equalivent to `::tensorflow::Node::IsControlFlow()`.
  // It requires to construct a `::tensorflow::Graph` to use that helper
  // function, so we simply hardcode the list of control flow ops here.
  if (tensorflow_op == "Switch" || tensorflow_op == "RefSwitch" ||
      tensorflow_op == "Merge" || tensorflow_op == "RefMerge" ||
      tensorflow_op == "Enter" || tensorflow_op == "RefEnter" ||
      tensorflow_op == "Exit" || tensorflow_op == "RefExit" ||
      tensorflow_op == "NextIteration" || tensorflow_op == "RefNextIteration") {
    return true;
  }
  // TODO(ycling): Also check how to handle Variable ops and Assign ops.
  return false;
}

// Map from operator name to TF Lite enum value, for all builtins.
const std::map<string, BuiltinOperator>& GetBuiltinOpsMap() {
  static std::map<string, BuiltinOperator>* builtin_ops = nullptr;
  if (builtin_ops == nullptr) {
    builtin_ops = new std::map<string, BuiltinOperator>();

    for (int i = BuiltinOperator_MIN; i <= BuiltinOperator_MAX; ++i) {
      BuiltinOperator op = static_cast<BuiltinOperator>(i);
      string name = EnumNameBuiltinOperator(op);
      if (op != BuiltinOperator_CUSTOM && !name.empty()) {
        (*builtin_ops)[name] = op;
      }
    }
  }
  return *builtin_ops;
}

void WriteModelToString(const flatbuffers::FlatBufferBuilder& builder,
                        string* file_contents) {
  const uint8_t* buffer = builder.GetBufferPointer();
  int size = builder.GetSize();
  *file_contents = string(reinterpret_cast<const char*>(buffer), size);
}

}  // Anonymous namespace.

namespace details {

OperatorKey GetOperatorKey(
    const ::toco::Operator& op,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    bool allow_flex_ops) {
  // Get the op name (by Toco definition).
  string name = HelpfulOperatorTypeName(op);

  bool is_builtin = false;
  OperatorKey key;

  const auto& builtin_ops = GetBuiltinOpsMap();
  if (ops_by_type.count(op.type) != 0) {
    key.version = ops_by_type.at(op.type)->GetVersion(op);
    name = ops_by_type.at(op.type)->name();
    is_builtin = (builtin_ops.count(name) > 0);
  }

  if (is_builtin) {
    // For TFLite supported builtin ops, find out its BuiltinOperator enum used
    // in FlatBuffer.
    key.type = builtin_ops.at(name);
    return key;
  }

  // The logic below is all for custom ops.
  key.is_custom_op = true;
  key.type = BuiltinOperator_CUSTOM;

  if (op.type == OperatorType::kUnsupported) {
    const TensorFlowUnsupportedOperator& unsupported_op =
        static_cast<const TensorFlowUnsupportedOperator&>(op);
    const auto tensorflow_op = unsupported_op.tensorflow_op;

    // TODO(b/113715895): When `allow_flex_ops` is on, for now there's no way
    // to populate a regular custom op. We need to find a way to fix this.
    if (allow_flex_ops) {
      key.is_flex_op = true;
      key.flex_tensorflow_op = tensorflow_op;
      key.custom_code =
          string(::tflite::kFlexCustomCodePrefix) + key.flex_tensorflow_op;
    } else {
      key.custom_code = tensorflow_op;
    }
  } else if (allow_flex_ops && !op.tensorflow_node_def.empty()) {
    // For Toco-supported/TFLite-unsupported ops, if the TensorFlow NodeDef
    // is retained in the Toco Operator, we produce a Flex op if Flex mode
    // is enabled.
    key.is_flex_op = true;
    key.flex_tensorflow_op = name;
    key.custom_code =
        string(::tflite::kFlexCustomCodePrefix) + key.flex_tensorflow_op;
  } else {
    // If Flex is disabled or the original TensorFlow NodeDef isn't available,
    // we produce a custom op. This gives developers a chance to implemenr
    // custom ops.
    key.custom_code = name;
  }

  if (key.is_flex_op) {
    if (IsControlFlowOp(key.flex_tensorflow_op)) {
      key.is_unsupported_flex_op = true;
    }
  }
  return key;
}

void LoadTensorsMap(const Model& model, TensorsMap* tensors_map) {
  // First find a list of unique array names.
  std::set<string> names;
  for (const auto& array_pair : model.GetArrayMap()) {
    names.insert(array_pair.first);
  }

  // Now assign indices to them and fill in the map.
  int index = 0;
  for (const auto& name : names) {
    (*tensors_map)[name] = index;
    ++index;
  }
}

void LoadOperatorsMap(
    const Model& model, OperatorsMap* operators_map,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    bool allow_flex_ops) {
  // First find a list of unique operator types.
  std::set<OperatorKey> keys;
  for (const auto& op : model.operators) {
    keys.insert(GetOperatorKey(*op, ops_by_type, allow_flex_ops));
  }
  // Now assign indices to them and fill in the map.
  int index = 0;
  for (const auto& key : keys) {
    (*operators_map)[key] = index;
    ++index;
  }
}

}  // namespace details

Offset<Vector<Offset<Tensor>>> ExportTensors(
    const Model& model, const details::TensorsMap& tensors_map,
    FlatBufferBuilder* builder, std::vector<const Array*>* buffers_to_write,
    const std::set<int32_t>& variable_tensor_indices) {
  // In the end we will need to produce a vector sorted by the indices of the
  // tensors in the tensors_map.
  std::map<int, Offset<Tensor>> ordered_tensors;

  for (const auto& array_pair : model.GetArrayMap()) {
    const string& tensor_name = array_pair.first;
    const toco::Array& array = *array_pair.second;

    int buffer_index = buffers_to_write->size();
    auto type = DataType::Serialize(array.data_type);
    buffers_to_write->push_back(&array);

    std::vector<int> shape;
    if (array.has_shape()) {
      for (int d : array.shape().dims()) {
        shape.push_back(d);
      }
    }

    Offset<Vector<float>> min;
    Offset<Vector<float>> max;
    Offset<Vector<float>> scale;
    Offset<Vector<int64_t>> zero_point;
    if (array.minmax) {
      min = builder->CreateVector(
          std::vector<float>{static_cast<float>(array.minmax->min)});
      max = builder->CreateVector(
          std::vector<float>{static_cast<float>(array.minmax->max)});
    }
    if (array.quantization_params) {
      scale = builder->CreateVector(std::vector<float>{
          static_cast<float>(array.quantization_params->scale)});
      zero_point = builder->CreateVector(
          std::vector<int64_t>{array.quantization_params->zero_point});
    }
    auto q_param = ::tflite::CreateQuantizationParameters(*builder, min, max,
                                                          scale, zero_point);

    int index = tensors_map.at(tensor_name);
    bool is_variable =
        variable_tensor_indices.find(index) != variable_tensor_indices.end();
    ordered_tensors[index] =
        CreateTensor(*builder, builder->CreateVector(shape), type, buffer_index,
                     builder->CreateString(tensor_name), q_param, is_variable);
  }

  std::vector<Offset<Tensor>> tensor_vector;
  tensor_vector.reserve(ordered_tensors.size());
  for (const auto& tensor : ordered_tensors) {
    tensor_vector.push_back(tensor.second);
  }

  return builder->CreateVector(tensor_vector);
}

Offset<Vector<int32_t>> ExportInputTensors(
    const Model& model, const details::TensorsMap& tensors_map,
    FlatBufferBuilder* builder) {
  std::vector<int32_t> inputs;
  for (const auto& input : model.flags.input_arrays()) {
    inputs.push_back(tensors_map.at(input.name()));
  }
  return builder->CreateVector<int32_t>(inputs);
}

Offset<Vector<int32_t>> ExportOutputTensors(
    const Model& model, const details::TensorsMap& tensors_map,
    FlatBufferBuilder* builder) {
  std::vector<int32_t> outputs;
  for (const string& output : model.flags.output_arrays()) {
    outputs.push_back(tensors_map.at(output));
  }
  return builder->CreateVector<int32_t>(outputs);
}

Offset<Vector<Offset<OperatorCode>>> ExportOperatorCodes(
    const Model& model,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    const details::OperatorsMap& operators_map, FlatBufferBuilder* builder,
    const ExportParams& params) {
  // Map from operator name to TF Lite enum value, for all builtins.
  std::map<string, BuiltinOperator> builtin_ops;
  for (int i = BuiltinOperator_MIN; i <= BuiltinOperator_MAX; ++i) {
    BuiltinOperator op = static_cast<BuiltinOperator>(i);
    string name = EnumNameBuiltinOperator(op);
    if (op != BuiltinOperator_CUSTOM && !name.empty()) {
      builtin_ops[name] = op;
    }
  }

  // We will need to produce a vector of codes in the same order as they
  // appear in the operators_map.
  std::map<int, Offset<OperatorCode>> ordered_opcodes;

  for (const auto& op : model.operators) {
    const details::OperatorKey operator_key =
        details::GetOperatorKey(*op, ops_by_type, params.allow_flex_ops);
    int op_index = operators_map.at(operator_key);

    flatbuffers::Offset<flatbuffers::String> custom_code = 0;
    if (!operator_key.custom_code.empty()) {
      custom_code = builder->CreateString(operator_key.custom_code);
    }

    ordered_opcodes[op_index] = CreateOperatorCode(
        *builder, operator_key.type, custom_code, operator_key.version);
  }

  std::vector<Offset<OperatorCode>> opcode_vector;
  opcode_vector.reserve(ordered_opcodes.size());
  for (const auto& opcode : ordered_opcodes) {
    opcode_vector.push_back(opcode.second);
  }

  return builder->CreateVector(opcode_vector);
}

Offset<Vector<Offset<Operator>>> ExportOperators(
    const Model& model,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    const details::OperatorsMap& operators_map,
    const details::TensorsMap& tensors_map, FlatBufferBuilder* builder,
    std::set<int32_t>* variable_tensor_indices, const ExportParams& params) {
  variable_tensor_indices->clear();

  // The operators are in execution order, so we just follow tf.mini order.
  std::vector<Offset<Operator>> op_vector;
  for (const auto& op : model.operators) {
    std::vector<int32_t> inputs;
    for (const string& input : op->inputs) {
      // -1 is the ID for optional tensor in TFLite output
      int id = model.IsOptionalArray(input) ? -1 : tensors_map.at(input);
      inputs.push_back(id);
    }
    std::vector<int32_t> outputs;
    for (const string& output : op->outputs) {
      outputs.push_back(tensors_map.at(output));
    }

    const auto key =
        details::GetOperatorKey(*op, ops_by_type, params.allow_flex_ops);
    int op_index = operators_map.at(key);

    auto tflite_op_it = ops_by_type.find(op->type);
    BaseOperator* tflite_op = tflite_op_it == ops_by_type.end()
                                  ? nullptr
                                  : tflite_op_it->second.get();

    // This is a custom op unless we can find it in ops_by_type, and even then
    // it could be a custom op (such as kUnsupported).
    auto options = Options::Custom(0);

    std::vector<bool> mutating_input_variables;
    if (tflite_op) {
      options = tflite_op->Serialize(*op, builder);
      mutating_input_variables = tflite_op->GetMutatingInputVariables(*op);

      if (!mutating_input_variables.empty()) {
        for (int i = 0; i < op->inputs.size(); ++i) {
          if (!mutating_input_variables[i]) {
            continue;
          }
          int32_t variable_tensor_index = tensors_map.at(op->inputs[i]);
          variable_tensor_indices->insert(variable_tensor_index);
        }
      }
    } else if (key.is_flex_op && !op->tensorflow_node_def.empty()) {
      auto fbb = WriteFlexOpOptions(op->tensorflow_node_def);
      if (fbb) {
        options = Options::Custom(builder->CreateVector(fbb->GetBuffer()));
      }
    }
    // The only supported CustomOptionFormat is FLEXBUFFERS now.
    op_vector.push_back(CreateOperator(
        *builder, op_index, builder->CreateVector(inputs),
        builder->CreateVector(outputs), options.type, options.builtin,
        options.custom, ::tflite::CustomOptionsFormat_FLEXBUFFERS,
        builder->CreateVector(mutating_input_variables)));
  }

  return builder->CreateVector(op_vector);
}

Offset<Vector<Offset<Buffer>>> ExportBuffers(
    const Model& model, const std::vector<const Array*>& buffers_to_write,
    FlatBufferBuilder* builder) {
  std::vector<Offset<Buffer>> buffer_vector;
  size_t index = 0;
  for (const Array* array_ptr : buffers_to_write) {
    const Array& array = *array_ptr;
    Offset<Vector<uint8_t>> data_buffer = DataBuffer::Serialize(array, builder);
    buffer_vector.push_back(CreateBuffer(*builder, data_buffer));
    index++;
  }
  return builder->CreateVector(buffer_vector);
}

void Export(const Model& model, string* output_file_contents,
            const ExportParams& params) {
  const auto ops_by_type = BuildOperatorByTypeMap(params.allow_flex_ops);
  Export(model, output_file_contents, params, ops_by_type);
}

void Export(
    const Model& model, string* output_file_contents,
    const ExportParams& params,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type) {
  flatbuffers::FlatBufferBuilder builder(/*initial_size=*/10240);

  details::TensorsMap tensors_map;
  details::LoadTensorsMap(model, &tensors_map);

  details::OperatorsMap operators_map;
  details::LoadOperatorsMap(model, &operators_map, ops_by_type,
                            params.allow_flex_ops);

  std::vector<const Array*> buffers_to_write;
  Array empty_array;
  buffers_to_write.push_back(&empty_array);

  auto op_codes =
      ExportOperatorCodes(model, ops_by_type, operators_map, &builder, params);

  for (const auto& op : model.operators) {
    if (op->type == OperatorType::kFakeQuant) {
      LOG(WARNING) << "FAKE_QUANT operation " << LogName(*op)
                   << " was not converted. If running quantized make sure you "
                      "are passing --inference_type=QUANTIZED_UINT8 and values "
                      "for --std_values and --mean_values.";
    }
  }

  std::set<string> custom_ops;
  std::set<string> unsupported_flex_ops;
  for (const auto& it : operators_map) {
    const details::OperatorKey& key = it.first;
    if (key.is_custom_op) {
      custom_ops.insert(key.custom_code);
    }
    if (key.is_unsupported_flex_op) {
      unsupported_flex_ops.insert(key.flex_tensorflow_op);
    }
  }

  if (!custom_ops.empty()) {
    if (!params.allow_custom_ops) {
      // Remove ExpandDims and ReorderAxes from unimplemented list unless they
      // compose the list. Both ops are removed during graph transformations.
      // However, if an op is unimplemented earlier in the model, the graph
      // transformation is unable to run because the output shape is not
      // defined. This causes unnecessary confusion during model conversion
      // time.
      std::set<string> custom_ops_final;
      for (const auto& op_type : custom_ops) {
        if (op_type != "ReorderAxes" && op_type != "ExpandDims") {
          custom_ops_final.insert(op_type);
        }
      }
      if (custom_ops_final.empty()) {
        custom_ops_final = custom_ops;
      }

      LOG(QFATAL)
          << "Some of the operators in the model are not supported by "
             "the standard TensorFlow Lite runtime. If you have a custom "
             "implementation for them you can disable this error with "
             "--allow_custom_ops, or by setting allow_custom_ops=True "
             "when calling tf.contrib.lite.TFLiteConverter(). Here is a list "
             "of operators for which  you will need custom implementations: "
          << absl::StrJoin(custom_ops_final, ", ") << ".";
    }

    std::set<string> unsupported_control_flow_ops;
    // Check if unsupported ops contains control flow ops. It's impossible
    // to implement these ops as custom ops at the moment.
    for (const auto& op : custom_ops) {
      if (IsControlFlowOp(op)) {
        unsupported_control_flow_ops.insert(op);
      }
    }
    if (!unsupported_control_flow_ops.empty()) {
      LOG(QFATAL)
          << "TensorFlow Lite currently doesn't support control flow ops: "
          << absl::StrJoin(unsupported_control_flow_ops, ", ") << ".";
    }
  }

  if (!unsupported_flex_ops.empty()) {
    LOG(QFATAL) << "Some of the operators in the model are not supported by "
                   "TensorFlow Flex runtime: "
                << absl::StrJoin(unsupported_flex_ops, ", ") << ".";
  }

  std::set<int32_t> variable_tensor_indices;
  auto ops = ExportOperators(model, ops_by_type, operators_map, tensors_map,
                             &builder, &variable_tensor_indices, params);

  auto tensors = ExportTensors(model, tensors_map, &builder, &buffers_to_write,
                               variable_tensor_indices);
  auto inputs = ExportInputTensors(model, tensors_map, &builder);
  auto outputs = ExportOutputTensors(model, tensors_map, &builder);

  // TODO(aselle): add support to toco for multiple subgraphs.
  auto subgraph = CreateSubGraph(builder, tensors, inputs, outputs, ops,
                                 /* name */ 0);
  std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {subgraph};

  auto buffers = ExportBuffers(model, buffers_to_write, &builder);
  auto description = builder.CreateString("TOCO Converted.");
  auto new_model_location =
      CreateModel(builder, TFLITE_SCHEMA_VERSION, op_codes,
                  builder.CreateVector(subgraphs), description, buffers);
  ::tflite::FinishModelBuffer(builder, new_model_location);

  if (params.quantize_weights) {
    // Call the quantize_weights tool.
    LOG(INFO) << "Quantizing TFLite model after conversion to flatbuffer. "
                 "dump_graphviz will only output the model before this "
                 "transformation. To visualize the output graph use "
                 "lite/tools/optimize.py.";
    flatbuffers::FlatBufferBuilder q_builder(/*initial_size=*/10240);
    const uint8_t* buffer = builder.GetBufferPointer();
    const ::tflite::Model* input_model = ::tflite::GetModel(buffer);
    if (::tflite::optimize::QuantizeWeights(&q_builder, input_model) !=
        kTfLiteOk) {
      LOG(QFATAL) << "Quantize weights transformation failed.";
    }
    WriteModelToString(q_builder, output_file_contents);
  } else {
    WriteModelToString(builder, output_file_contents);
  }
}

}  // namespace tflite

}  // namespace toco
