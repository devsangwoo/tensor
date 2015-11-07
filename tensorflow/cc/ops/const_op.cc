<<<<<<< HEAD
/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/types.h"
=======
#include "tensorflow/cc/ops/const_op.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {
namespace ops {

namespace {
<<<<<<< HEAD
template <typename T>
Output ConstHelper(const Scope& scope, const T& value, DataType dtype) {
  if (!scope.ok()) return Output();

  Node* ret;
  Graph* graph = scope.graph();
  const string unique_name = scope.GetUniqueNameForOp("Const");
  auto builder = NodeBuilder(unique_name, "Const")
                     .Attr("value", value)
                     .Attr("dtype", dtype);
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(graph, &ret));
  if (!scope.ok()) return Output();

  scope.UpdateStatus(scope.DoShapeInference(ret));
  if (!scope.ok()) return Output();

  return Output(ret);
}
}  // namespace

Output Const(const Scope& scope, const Input::Initializer& val) {
  if (!val.status.ok()) {
    scope.UpdateStatus(val.status);
    return Output();
  }
  return ConstHelper(scope, val.tensor, val.tensor.dtype());
}

Output ConstFromProto(const Scope& scope, const TensorProto& proto) {
  return ConstHelper(scope, proto, proto.dtype());
}

NodeBuilder::NodeOut AsNodeOut(const Scope& scope, const Input& inp) {
  if (!inp.status().ok()) {
    scope.UpdateStatus(inp.status());
    return NodeBuilder::NodeOut(inp.node(), inp.index());
  }
  if (inp.node()) {
    return NodeBuilder::NodeOut(inp.node(), inp.index());
  }
  if (!inp.node_name().empty()) {
    return NodeBuilder::NodeOut(inp.node_name(), inp.index(), inp.data_type());
  }
  auto transformed = Input{
      Const(scope.NewSubScope("Const"), Input::Initializer(inp.tensor()))};
  return NodeBuilder::NodeOut{transformed.node(), transformed.index()};
}

std::vector<NodeBuilder::NodeOut> AsNodeOutList(const Scope& scope,
                                                const InputList& inp) {
  std::vector<NodeBuilder::NodeOut> out;
  for (const auto& i : inp) {
    const auto node_out = AsNodeOut(scope, i);
    if (!scope.ok()) {
      return {};
    }
    out.push_back(node_out);
  }
  return out;
=======
const string& OpName() {
  static const string kOpName = "Const";
  return kOpName;
}
}  // namespace

#define DEFINE_CONST_SCALAR(TYPE)                                         \
  Node* Const(TYPE s, const GraphDefBuilder::Options& options) {          \
    return Const(gtl::ArraySlice<TYPE>(&s, 1), TensorShape({}), options); \
  }

#define DEFINE_CONST_VECTOR(TYPE)                                          \
  Node* Const(gtl::ArraySlice<TYPE> v,                                     \
              const GraphDefBuilder::Options& options) {                   \
    return Const(v, TensorShape({static_cast<int64>(v.size())}), options); \
  }

#define DEFINE_CONST_TENSOR(TYPE, ...)                                         \
  Node* Const(gtl::ArraySlice<TYPE> t, const TensorShape& shape,               \
              const GraphDefBuilder::Options& options) {                       \
    if (options.HaveError()) return nullptr;                                   \
    NodeBuilder node_builder(options.GetNameForOp(OpName()), OpName(),         \
                             options.op_registry());                           \
    const DataType dt = DataTypeToEnum<TYPE>::v();                             \
    if (t.size() == 1) {                                                       \
      TensorProto proto;                                                       \
      proto.set_dtype(dt);                                                     \
      shape.AsProto(proto.mutable_tensor_shape());                             \
      __VA_ARGS__;                                                             \
      node_builder.Attr("dtype", dt).Attr("value", proto);                     \
    } else {                                                                   \
      Tensor tensor(dt, shape);                                                \
      if (tensor.NumElements() != static_cast<int64>(t.size())) {              \
        options.UpdateStatus(errors::InvalidArgument(                          \
            t.size(), " values provided to Const() != ", tensor.NumElements(), \
            " elements for shape ", shape.ShortDebugString()));                \
      } else {                                                                 \
        std::copy_n(t.data(), t.size(), tensor.flat<TYPE>().data());           \
        node_builder.Attr("dtype", dt).Attr("value", tensor);                  \
      }                                                                        \
    }                                                                          \
    return options.FinalizeBuilder(&node_builder);                             \
  }

#define DEFINE_CONST_IMPL(TYPE, ...) \
  DEFINE_CONST_SCALAR(TYPE)          \
  DEFINE_CONST_VECTOR(TYPE)          \
  DEFINE_CONST_TENSOR(TYPE, __VA_ARGS__)

#define DEFINE_CONST(TYPE, FIELD) \
  DEFINE_CONST_IMPL(TYPE, proto.add_##FIELD(*t.begin());)

DEFINE_CONST(float, float_val);
DEFINE_CONST(double, double_val);
DEFINE_CONST(int32, int_val);
DEFINE_CONST(uint8, int_val);
DEFINE_CONST(int16, int_val);
DEFINE_CONST(int8, int_val);
DEFINE_CONST(int64, int64_val);
DEFINE_CONST(bool, bool_val);

DEFINE_CONST_IMPL(complex64, proto.add_scomplex_val(t.begin()->real());
                  proto.add_scomplex_val(t.begin()->imag()););

Node* Const(StringPiece s, const GraphDefBuilder::Options& options) {
  if (options.HaveError()) return nullptr;
  NodeBuilder node_builder(options.GetNameForOp(OpName()), OpName(),
                           options.op_registry());
  TensorProto proto;
  proto.set_dtype(DT_STRING);
  TensorShape({}).AsProto(proto.mutable_tensor_shape());
  proto.add_string_val(s.data(), s.size());
  node_builder.Attr("dtype", DT_STRING).Attr("value", proto);
  return options.FinalizeBuilder(&node_builder);
}

DEFINE_CONST_VECTOR(string)
DEFINE_CONST_TENSOR(string, proto.add_string_val(*t.begin());)

#undef DEFINE_CONST
#undef DEFINE_CONST_IMPL
#undef DEFINE_CONST_TENSOR
#undef DEFINE_CONST_VECTOR
#undef DEFINE_CONST_SCALAR

Node* Const(const Tensor& t, const GraphDefBuilder::Options& options) {
  if (options.HaveError()) return nullptr;
  NodeBuilder node_builder(options.GetNameForOp(OpName()), OpName(),
                           options.op_registry());
  node_builder.Attr("dtype", t.dtype()).Attr("value", t);
  return options.FinalizeBuilder(&node_builder);
}

Node* Const(const TensorProto& proto, const GraphDefBuilder::Options& options) {
  if (options.HaveError()) return nullptr;
  NodeBuilder node_builder(options.GetNameForOp(OpName()), OpName(),
                           options.op_registry());
  node_builder.Attr("dtype", proto.dtype()).Attr("value", proto);
  return options.FinalizeBuilder(&node_builder);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

}  // namespace ops
}  // namespace tensorflow
