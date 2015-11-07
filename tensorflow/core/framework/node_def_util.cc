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

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/framework/node_def_util.h"

#include <algorithm>
#include <unordered_map>
<<<<<<< HEAD
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

const char* const kColocationAttrName = "_class";
const char* const kColocationGroupPrefix = "loc:@";

AttrSlice::AttrSlice() : ndef_(nullptr) {
  static const AttrValueMap* const kEmptyAttrValueMap = new AttrValueMap;
  attrs_ = kEmptyAttrValueMap;
}

AttrSlice::AttrSlice(const NodeDef& node_def)
    : ndef_(&node_def), attrs_(&ndef_->attr()) {}

AttrSlice::AttrSlice(const AttrValueMap* a) : ndef_(nullptr), attrs_(a) {}

string SummarizeAttrsHelper(AttrSlice attrs, StringPiece device) {
  string ret;

  // We sort the attrs so the output is deterministic.
  std::vector<string> attr_names;
  attr_names.reserve(attrs.size());
  for (const auto& attr : attrs) {
=======

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {

string SummarizeNodeDef(const NodeDef& node_def) {
  string ret = strings::StrCat(node_def.name(), " = ", node_def.op(), "[");

  // We sort the attrs so the output is deterministic.
  std::vector<string> attr_names;
  attr_names.reserve(node_def.attr().size());
  for (const auto& attr : node_def.attr()) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    attr_names.push_back(attr.first);
  }
  std::sort(attr_names.begin(), attr_names.end());
  bool first = true;
  for (const string& attr_name : attr_names) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
<<<<<<< HEAD
    strings::StrAppend(&ret, attr_name, "=",
                       SummarizeAttrValue(*attrs.Find(attr_name)));
  }

  // Consider the device to be a final attr with name "_device".
  if (!device.empty()) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
    strings::StrAppend(&ret, "_device=\"", device, "\"");
  }
  return ret;
}

string AttrSlice::SummarizeNode() const {
  return ndef_ ? SummarizeNodeDef(*ndef_)
               : strings::StrCat(
                     "[", SummarizeAttrsHelper(*this, StringPiece()), "]");
}

string AttrSlice::DebugString() const {
  std::vector<string> attr_key_vals;
  attr_key_vals.reserve(attrs_->size());
  for (const auto& it : *this) {
    const string& name = it.first;
    const AttrValue& attr_value = it.second;
    attr_key_vals.push_back(
        absl::StrCat(name, "=", SummarizeAttrValue(attr_value)));
  }
  return absl::StrJoin(attr_key_vals, ", ");
}

string SummarizeNodeDef(const NodeDef& node_def) {
  string ret = strings::StrCat(errors::FormatNodeNameForError(node_def.name()),
                               " = ", node_def.op(), "[");
  strings::StrAppend(&ret, SummarizeAttrsHelper(node_def, node_def.device()));
  strings::StrAppend(&ret, "](");

  // Output inputs, including control inputs, verbatim.
  bool first = true;
=======
    auto iter = node_def.attr().find(attr_name);
    strings::StrAppend(&ret, attr_name, "=", SummarizeAttrValue(iter->second));
  }

  // Consider the device to be a final attr with name "_device".
  if (!node_def.device().empty()) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
    strings::StrAppend(&ret, "_device=\"", node_def.device(), "\"");
  }
  strings::StrAppend(&ret, "](");

  // Output inputs, including control inputs, verbatim.
  first = true;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  for (const string& input : node_def.input()) {
    if (!first) strings::StrAppend(&ret, ", ");
    first = false;
    strings::StrAppend(&ret, input);
  }
  strings::StrAppend(&ret, ")");
  return ret;
}

<<<<<<< HEAD
string SummarizeAttrs(const NodeDef& node_def) {
  return SummarizeAttrsHelper(node_def, node_def.device());
}

string FormatNodeDefForError(
    StringPiece node_name, bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info) {
  return !has_experimental_debug_info ||
                 experimental_debug_info.original_node_names().empty()
             ? errors::FormatNodeNameForError(string(node_name))
             : errors::FormatNodeNamesForError(
                   experimental_debug_info.original_node_names());
}

string FormatNodeDefForError(const NodeDef& node_def) {
  return FormatNodeDefForError(node_def.name(),
                               node_def.has_experimental_debug_info(),
                               node_def.experimental_debug_info());
}

const AttrValue* AttrSlice::Find(StringPiece attr_name) const {
  // Currently, the collection used for NodeDef::attr() (google::protobuf::Map)
  // requires that the keys used for lookups have type 'const string&'. Because
  // this method takes a StringPiece, it is necessary to allocate a temporary
  // string, copy attr_name to it, and then use that temporary string for the
  // lookup. This causes an excessive number of short-lived allocations, and for
  // large graphs, this can be a significant cost.
  //
  // Because most nodes have a small number of attributes, a simple linear scan
  // is generally more efficient than a hashed lookup.  If google::protobuf::Map
  // changes so that it supports efficient lookups using StringPiece instead of
  // const string&, then this code could be changed to use attrs_->find() again.

  for (const auto& attr : *attrs_) {
    if (attr.first == attr_name) {
      return &attr.second;
    }
  }
  return nullptr;
}

Status AttrSlice::Find(StringPiece attr_name,
=======
const AttrValue* AttrSlice::Find(const string& attr_name) const {
  auto iter = attrs_->find(attr_name);
  if (iter == attrs_->end()) return nullptr;
  return &iter->second;
}

Status AttrSlice::Find(const string& attr_name,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                       const AttrValue** attr_value) const {
  *attr_value = Find(attr_name);
  if (*attr_value != nullptr) {
    return Status::OK();
  }
  Status s = errors::NotFound("No attr named '", attr_name, "' in NodeDef:");
<<<<<<< HEAD
  // Skip AttachDef for internal attrs since it is a little bit
  // expensive and it is common for them to correctly not be included
  // in a NodeDef.
  if (!absl::StartsWith(attr_name, "_") && ndef_ != nullptr) {
=======
  if (ndef_) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    s = AttachDef(s, *ndef_);
  }
  return s;
}

<<<<<<< HEAD
bool AttrSlice::EqualAttrs(AttrSlice other, Scratch* scratch) const {
  if (size() != other.size()) return false;

  for (const auto& attr : *other.attrs_) {
    auto iter = attrs_->find(attr.first);
    if (iter == attrs_->end()) return false;
    // TODO(irving): Comparing AttrValues by proto is slightly buggy, since
    // TensorProto is a nonunique representation of Tensor.  This bug will go
    // away once AttrSlice switches over to NodeInfo.
    iter->second.SerializeToString(&scratch->a);
    attr.second.SerializeToString(&scratch->b);
    if (scratch->a != scratch->b) return false;
  }
  return true;
}

// The ... is to allow the caller to inject some value validation code.  Use
// just ; if no additional validation code is needed.
#define DEFINE_GET_ATTR(TYPE, FIELD, ATTR_TYPE, APPEND_OP, CAST, ...)         \
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,           \
=======
// The ... is to allow the caller to inject some value validation code.  Use
// just ; if no additional validation code is needed.
#define DEFINE_GET_ATTR(TYPE, FIELD, ATTR_TYPE, APPEND_OP, CAST, ...)         \
  Status GetNodeAttr(const AttrSlice& attrs, const string& attr_name,         \
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                     TYPE* value) {                                           \
    const AttrValue* attr_value;                                              \
    TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));                   \
    TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, ATTR_TYPE));             \
    const auto& v = attr_value->FIELD();                                      \
    __VA_ARGS__;                                                              \
    *value = CAST;                                                            \
    return Status::OK();                                                      \
  }                                                                           \
<<<<<<< HEAD
  Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,           \
=======
  Status GetNodeAttr(const AttrSlice& attrs, const string& attr_name,         \
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                     std::vector<TYPE>* value) {                              \
    const AttrValue* attr_value;                                              \
    TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));                   \
    TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "list(" ATTR_TYPE ")")); \
<<<<<<< HEAD
    value->reserve(attr_value->list().FIELD().size());                        \
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    for (const auto& v : attr_value->list().FIELD()) {                        \
      __VA_ARGS__;                                                            \
      value->APPEND_OP(CAST);                                                 \
    }                                                                         \
    return Status::OK();                                                      \
  }

<<<<<<< HEAD
#define DEFINE_TRY_GET_ATTR(TYPE, FIELD, ATTR_TYPE, APPEND_OP, CAST, ...) \
  bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,      \
                      TYPE* value) {                                      \
    const AttrValue* attr_value = attrs.Find(attr_name);                  \
    if (attr_value == nullptr) {                                          \
      return false;                                                       \
    }                                                                     \
    Status s = AttrValueHasType(*attr_value, ATTR_TYPE);                  \
    if (!s.ok()) {                                                        \
      return false;                                                       \
    }                                                                     \
    const auto& v = attr_value->FIELD();                                  \
    __VA_ARGS__;                                                          \
    *value = CAST;                                                        \
    return true;                                                          \
  }                                                                       \
  bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,      \
                      std::vector<TYPE>* value) {                         \
    const AttrValue* attr_value = attrs.Find(attr_name);                  \
    if (attr_value == nullptr) {                                          \
      return false;                                                       \
    }                                                                     \
    Status s = AttrValueHasType(*attr_value, "list(" ATTR_TYPE ")");      \
    if (!s.ok()) {                                                        \
      return false;                                                       \
    }                                                                     \
    value->reserve(attr_value->list().FIELD().size());                    \
    for (const auto& v : attr_value->list().FIELD()) {                    \
      __VA_ARGS__;                                                        \
      value->APPEND_OP(CAST);                                             \
    }                                                                     \
    return true;                                                          \
  }
#ifdef USE_TSTRING
DEFINE_GET_ATTR(tstring, s, "string", emplace_back, v, ;)
DEFINE_TRY_GET_ATTR(tstring, s, "string", emplace_back, v, ;)
#endif
DEFINE_GET_ATTR(string, s, "string", emplace_back, v, ;)
DEFINE_TRY_GET_ATTR(string, s, "string", emplace_back, v, ;)
DEFINE_GET_ATTR(int64, i, "int", emplace_back, v, ;)
DEFINE_TRY_GET_ATTR(int64, i, "int", emplace_back, v, ;)
DEFINE_GET_ATTR(
    int32, i, "int", emplace_back, static_cast<int32>(v),
    if (static_cast<int64>(static_cast<int32>(v)) != v) {
      return errors::InvalidArgument("Attr ", attr_name, " has value ", v,
                                     " out of range for an int32");
    })
DEFINE_TRY_GET_ATTR(
    int32, i, "int", emplace_back, static_cast<int32>(v),
    if (static_cast<int64>(static_cast<int32>(v)) != v) {
      static int log_counter = 0;
      if (log_counter < 10) {
        log_counter++;
        LOG(WARNING) << "Attr " << attr_name << " has value " << v
                     << " out of range for an int32";
      }
      return false;
    })
DEFINE_GET_ATTR(float, f, "float", emplace_back, v, ;)
DEFINE_TRY_GET_ATTR(float, f, "float", emplace_back, v, ;)
=======
DEFINE_GET_ATTR(string, s, "string", emplace_back, v, ;)
DEFINE_GET_ATTR(int64, i, "int", emplace_back, v, ;)
DEFINE_GET_ATTR(int32, i, "int", emplace_back, static_cast<int32>(v),
                if (static_cast<int64>(static_cast<int32>(v)) != v) {
                  return errors::InvalidArgument("Attr ", attr_name,
                                                 " has value ", v,
                                                 " out of range for an int32");
                })
DEFINE_GET_ATTR(float, f, "float", emplace_back, v, ;)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// std::vector<bool> specialization does not have emplace_back until
// c++14, so we have to use push_back (see
// http://en.cppreference.com/w/cpp/container/vector/emplace_back)
DEFINE_GET_ATTR(bool, b, "bool", push_back, v, ;)
<<<<<<< HEAD
DEFINE_TRY_GET_ATTR(bool, b, "bool", push_back, v, ;)
DEFINE_GET_ATTR(DataType, type, "type", emplace_back, static_cast<DataType>(v),
                ;)
DEFINE_TRY_GET_ATTR(DataType, type, "type", emplace_back,
                    static_cast<DataType>(v),
                    ;)
DEFINE_GET_ATTR(TensorShapeProto, shape, "shape", emplace_back, v, ;)
DEFINE_GET_ATTR(TensorShape, shape, "shape", emplace_back, TensorShape(v),
                TF_RETURN_IF_ERROR(TensorShape::IsValidShape(v));)
DEFINE_TRY_GET_ATTR(
    TensorShape, shape, "shape", emplace_back, TensorShape(v),
    if (!TensorShape::IsValidShape(v).ok()) {
      static int log_counter = 0;
      if (log_counter < 10) {
        log_counter++;
        LOG(WARNING) << "Attr " << attr_name << " has invalid shape value "
                     << v.DebugString();
      }
      return false;
    })
DEFINE_GET_ATTR(PartialTensorShape, shape, "shape", emplace_back,
                PartialTensorShape(v),
                TF_RETURN_IF_ERROR(PartialTensorShape::IsValidShape(v));)
DEFINE_GET_ATTR(
    Tensor, tensor, "tensor", emplace_back, t, Tensor t; if (!t.FromProto(v)) {
      return errors::InvalidArgument("Attr ", attr_name, " has value ",
                                     v.ShortDebugString(),
                                     " that can't be converted to a Tensor");
    })
DEFINE_GET_ATTR(NameAttrList, func, "func", emplace_back, v, ;);
#undef DEFINE_GET_ATTR

bool HasNodeAttr(const NodeDef& node_def, StringPiece attr_name) {
  return node_def.attr().find(string(attr_name)) != node_def.attr().end();
}

static const string& kEmptyString = *new string();

const string& GetNodeAttrString(const AttrSlice& attrs, StringPiece attr_name) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return kEmptyString;
  }
  Status s = AttrValueHasType(*attr_value, "string");
  if (!s.ok()) {
    return kEmptyString;
  }
  return attr_value->s();
}

bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<const string*>* value) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return false;
  }
  Status s = AttrValueHasType(*attr_value, "list(string)");
  if (!s.ok()) {
    return false;
  }
  value->reserve(attr_value->list().s().size());
  for (const auto& v : attr_value->list().s()) {
    value->push_back(&v);
  }
  return true;
}

bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    std::vector<const TensorShapeProto*>* value) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return false;
  }
  Status s = AttrValueHasType(*attr_value, "list(shape)");
  if (!s.ok()) {
    return false;
  }
  value->reserve(attr_value->list().shape().size());
  for (const auto& v : attr_value->list().shape()) {
    value->push_back(&v);
  }
  return true;
}

Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
=======
DEFINE_GET_ATTR(DataType, type, "type", emplace_back, static_cast<DataType>(v),
                ;)
DEFINE_GET_ATTR(TensorShapeProto, shape, "shape", emplace_back, v, ;)
DEFINE_GET_ATTR(TensorShape, shape, "shape", emplace_back, TensorShape(v), ;)
DEFINE_GET_ATTR(Tensor, tensor, "tensor", emplace_back, t, Tensor t;
                if (!t.FromProto(v)) {
                  return errors::InvalidArgument(
                      "Attr ", attr_name, " has value ", v.ShortDebugString(),
                      " that can't be converted to a Tensor");
                })

#undef DEFINE_GET_ATTR

Status GetNodeAttr(const AttrSlice& attrs, const string& attr_name,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                   DataTypeVector* value) {
  const AttrValue* attr_value;
  TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
  TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "list(type)"));
  for (const auto& v : attr_value->list().type()) {
    value->push_back(static_cast<DataType>(v));
  }
  return Status::OK();
}

<<<<<<< HEAD
Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
=======
Status GetNodeAttr(const AttrSlice& attrs, const string& attr_name,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                   const TensorProto** value) {
  const AttrValue* attr_value;
  TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
  TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "tensor"));
  *value = &attr_value->tensor();
  return Status::OK();
}

<<<<<<< HEAD
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    const TensorProto** value) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return false;
  }
  Status s = AttrValueHasType(*attr_value, "tensor");
  if (!s.ok()) {
    return false;
  }
  *value = &attr_value->tensor();
  return true;
}

Status GetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
=======
Status GetNodeAttr(const AttrSlice& attrs, const string& attr_name,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                   const NameAttrList** value) {
  const AttrValue* attr_value;
  TF_RETURN_IF_ERROR(attrs.Find(attr_name, &attr_value));
  TF_RETURN_IF_ERROR(AttrValueHasType(*attr_value, "func"));
  *value = &attr_value->func();
  return Status::OK();
}

<<<<<<< HEAD
bool TryGetNodeAttr(const AttrSlice& attrs, StringPiece attr_name,
                    const NameAttrList** value) {
  const AttrValue* attr_value = attrs.Find(attr_name);
  if (attr_value == nullptr) {
    return false;
  }
  Status s = AttrValueHasType(*attr_value, "func");
  if (!s.ok()) {
    return false;
  }
  *value = &attr_value->func();
  return true;
}

namespace {  // Helper for InOutTypesForNode().

template <class NodeDefOrAttrSlice>
Status AddArgToSig(const NodeDefOrAttrSlice& node_or_attrs,
                   const OpDef::ArgDef& arg_def, DataTypeVector* sig) {
=======
namespace {  // Helper for InOutTypesForNode().

Status AddArgToSig(const NodeDef& node_def, const OpDef::ArgDef& arg_def,
                   DataTypeVector* sig) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  const int original_size = sig->size();
  if (!arg_def.number_attr().empty()) {
    // Same type repeated "repeats" times.
    int32 repeats = -1;
<<<<<<< HEAD
    TF_RETURN_IF_ERROR(
        GetNodeAttr(node_or_attrs, arg_def.number_attr(), &repeats));
=======
    TF_RETURN_IF_ERROR(GetNodeAttr(node_def, arg_def.number_attr(), &repeats));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    if (repeats < 0) {
      return errors::InvalidArgument("Value for number_attr() ", repeats,
                                     " < 0");
    }

    if (!arg_def.type_attr().empty()) {
      DataType dtype;
<<<<<<< HEAD
      TF_RETURN_IF_ERROR(
          GetNodeAttr(node_or_attrs, arg_def.type_attr(), &dtype));
=======
      TF_RETURN_IF_ERROR(GetNodeAttr(node_def, arg_def.type_attr(), &dtype));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      for (int i = 0; i < repeats; ++i) {
        sig->push_back(dtype);
      }
    } else if (arg_def.type() != DT_INVALID) {
      for (int i = 0; i < repeats; ++i) {
        sig->push_back(arg_def.type());
      }
    } else {
      return errors::InvalidArgument("Missing type or type_attr field in ",
                                     arg_def.ShortDebugString());
    }
  } else if (!arg_def.type_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(
<<<<<<< HEAD
        AttrSlice(node_or_attrs).Find(arg_def.type_attr(), &attr_value));
=======
        AttrSlice(node_def).Find(arg_def.type_attr(), &attr_value));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    sig->push_back(attr_value->type());
  } else if (!arg_def.type_list_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(
<<<<<<< HEAD
        AttrSlice(node_or_attrs).Find(arg_def.type_list_attr(), &attr_value));
=======
        AttrSlice(node_def).Find(arg_def.type_list_attr(), &attr_value));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    for (int dtype : attr_value->list().type()) {
      sig->push_back(static_cast<DataType>(dtype));
    }
  } else if (arg_def.type() != DT_INVALID) {
    sig->push_back(arg_def.type());
  } else {
    return errors::InvalidArgument("No type fields in ",
                                   arg_def.ShortDebugString());
  }
  if (arg_def.is_ref()) {
    // For all types that were added by this function call, make them refs.
    for (size_t i = original_size; i < sig->size(); ++i) {
      (*sig)[i] = MakeRefType((*sig)[i]);
    }
  }
  return Status::OK();
}

}  // namespace

<<<<<<< HEAD
Status InputTypeForNode(const NodeDef& node_def, const OpDef& op_def,
                        int input_port, DataType* input_type) {
  DataTypeVector input_types;
  for (const auto& arg : op_def.input_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, &input_types));
    if (input_types.size() > input_port) {
      const DataType dtype = input_types[input_port];
      *input_type = dtype;
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Input ", input_port, " not found for node ",
                                 node_def.name());
}

Status InputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                         DataTypeVector* inputs) {
  for (const auto& arg : op_def.input_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, inputs));
  }
  return Status::OK();
}

Status OutputTypeForNode(const NodeDef& node_def, const OpDef& op_def,
                         int output_port, DataType* output_type) {
  DataTypeVector output_types;
  for (const auto& arg : op_def.output_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, &output_types));
    if (output_types.size() > output_port) {
      const DataType dtype = output_types[output_port];
      *output_type = dtype;
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Output ", output_port, " not found for node ",
                                 node_def.name());
}

Status OutputTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                          DataTypeVector* outputs) {
=======
Status InOutTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                         DataTypeVector* inputs, DataTypeVector* outputs) {
  for (const auto& arg : op_def.input_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, inputs));
  }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  for (const auto& arg : op_def.output_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(node_def, arg, outputs));
  }
  return Status::OK();
}

<<<<<<< HEAD
Status OutputTypesForNode(const AttrSlice& attrs, const OpDef& op_def,
                          DataTypeVector* outputs) {
  for (const auto& arg : op_def.output_arg()) {
    TF_RETURN_IF_ERROR(AddArgToSig(attrs, arg, outputs));
  }
  return Status::OK();
}

Status InOutTypesForNode(const NodeDef& node_def, const OpDef& op_def,
                         DataTypeVector* inputs, DataTypeVector* outputs) {
  TF_RETURN_IF_ERROR(InputTypesForNode(node_def, op_def, inputs));
  return OutputTypesForNode(node_def, op_def, outputs);
}

Status NumOutputsForNode(const NodeDef& node_def, const OpDef& op_def,
                         int* num_outputs) {
  DataTypeVector outputs;
  TF_RETURN_IF_ERROR(OutputTypesForNode(node_def, op_def, &outputs));
  *num_outputs = outputs.size();
  return Status::OK();
}

Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def) {
  if (node_def.op() != op_def.name()) {
    return errors::InvalidArgument(
        "NodeDef op '", node_def.op(), "' does not match ",
        SummarizeOpDef(op_def), "; NodeDef: ", FormatNodeDefForError(node_def));
=======
Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def) {
  if (node_def.op() != op_def.name()) {
    return errors::InvalidArgument("NodeDef op '", node_def.op(),
                                   "' does not match ", SummarizeOpDef(op_def),
                                   "; NodeDef: ", SummarizeNodeDef(node_def));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  bool seen_control = false;
  size_t num_inputs = 0;
  // TODO(josh11b): Unify the input field validation.
  for (const string& input : node_def.input()) {
<<<<<<< HEAD
    if (absl::StartsWith(input, "^")) {
=======
    if (StringPiece(input).starts_with("^")) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      seen_control = true;
      if (input.find(':') != string::npos) {
        return errors::InvalidArgument("Control input '", input,
                                       "' must not have ':' in NodeDef: ",
<<<<<<< HEAD
                                       FormatNodeDefForError(node_def));
=======
                                       SummarizeNodeDef(node_def));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      }
    } else if (seen_control) {
      return errors::InvalidArgument("Non-control input '", input,
                                     "' after control input in NodeDef: ",
<<<<<<< HEAD
                                     FormatNodeDefForError(node_def));
=======
                                     SummarizeNodeDef(node_def));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    } else {
      ++num_inputs;
    }
  }

  std::unordered_map<string, const OpDef::AttrDef*> op_attrs;
  for (const auto& attr : op_def.attr()) {
    if (!gtl::InsertIfNotPresent(&op_attrs, attr.name(), &attr)) {
      return errors::InvalidArgument("OpDef has duplicate attr name '",
<<<<<<< HEAD
                                     attr.name(),
                                     "': ", SummarizeOpDef(op_def));
=======
                                     attr.name(), "': ",
                                     SummarizeOpDef(op_def));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    }
  }
  for (const auto& attr : node_def.attr()) {
    // Allow internal optional attributes with names starting with "_".
<<<<<<< HEAD
    if (absl::StartsWith(attr.first, "_")) {
=======
    if (StringPiece(attr.first).starts_with("_")) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      continue;
    }
    auto iter = op_attrs.find(attr.first);
    if (iter == op_attrs.end()) {
<<<<<<< HEAD
      // A common cause of this error is that TensorFlow has made a
      // backwards-compatible change to the NodeDef (e.g., adding a
      // new attr with a default value), but the binary consuming the
      // NodeDef does not know about the new attribute; the solution
      // in these cases is to ensure that the binary consuming the
      // NodeDef is built with a version of TensorFlow no earlier than
      // the binary producing it.
      return errors::InvalidArgument(
          "NodeDef mentions attr '", attr.first, "' not in ",
          SummarizeOpDef(op_def),
          "; NodeDef: ", FormatNodeDefForError(node_def),
          ". (Check whether your GraphDef-interpreting binary is up to date "
          "with your GraphDef-generating binary.).");
    }
    // If attr value is placeholder, do not check it.
    if (attr.second.placeholder().empty()) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          ValidateAttrValue(attr.second, *iter->second),
          "; NodeDef: ", FormatNodeDefForError(node_def), "; ",
          SummarizeOpDef(op_def));
    }
=======
      return errors::InvalidArgument("NodeDef mentions attr '", attr.first,
                                     "' not in ", SummarizeOpDef(op_def),
                                     "; NodeDef: ", SummarizeNodeDef(node_def));
    }
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        ValidateAttrValue(attr.second, *iter->second), "; NodeDef: ",
        SummarizeNodeDef(node_def), "; ", SummarizeOpDef(op_def));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    // Keep track of which attr names have (not) been found in the NodeDef.
    op_attrs.erase(iter);
  }

  // Were all attrs in the OpDef found in the NodeDef?
  if (!op_attrs.empty()) {
    string attrs;
    for (const auto& attr_pair : op_attrs) {
      if (!attrs.empty()) strings::StrAppend(&attrs, "', '");
      strings::StrAppend(&attrs, attr_pair.first);
    }
<<<<<<< HEAD
    return errors::InvalidArgument(
        "NodeDef missing attr", op_attrs.size() == 1 ? " '" : "s '", attrs,
        "' from ", SummarizeOpDef(op_def),
        "; NodeDef: ", FormatNodeDefForError(node_def));
=======
    return errors::InvalidArgument("NodeDef missing attr",
                                   op_attrs.size() == 1 ? " '" : "s '", attrs,
                                   "' from ", SummarizeOpDef(op_def),
                                   "; NodeDef: ", SummarizeNodeDef(node_def));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  // Validate the number of inputs.
  DataTypeVector inputs, outputs;
  TF_RETURN_IF_ERROR(InOutTypesForNode(node_def, op_def, &inputs, &outputs));

  if (num_inputs != inputs.size()) {
    return errors::InvalidArgument(
        "NodeDef expected inputs '", DataTypeVectorString(inputs),
        "' do not match ", num_inputs, " inputs specified; ",
<<<<<<< HEAD
        SummarizeOpDef(op_def), "; NodeDef: ", FormatNodeDefForError(node_def));
=======
        SummarizeOpDef(op_def), "; NodeDef: ", SummarizeNodeDef(node_def));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  return Status::OK();
}

namespace {  // Helpers for NameRangesForNode()

<<<<<<< HEAD
Status ComputeArgRange(const AttrSlice& attrs, const OpDef::ArgDef& arg_def,
                       const OpDef& op_def, int* num) {
  if (!arg_def.number_attr().empty()) {
    // Same type repeated "num" times.
    return GetNodeAttr(attrs, arg_def.number_attr(), num);
  } else if (!arg_def.type_list_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(attrs.Find(arg_def.type_list_attr(), &attr_value));
=======
Status ComputeArgRange(const NodeDef& node_def, const OpDef::ArgDef& arg_def,
                       const OpDef& op_def, int* num) {
  if (!arg_def.number_attr().empty()) {
    // Same type repeated "num" times.
    return GetNodeAttr(node_def, arg_def.number_attr(), num);
  } else if (!arg_def.type_list_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(
        AttrSlice(node_def).Find(arg_def.type_list_attr(), &attr_value));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    *num = attr_value->list().type_size();
  } else if (!arg_def.type_attr().empty() || arg_def.type() != DT_INVALID) {
    *num = 1;
  } else {
<<<<<<< HEAD
    return errors::InvalidArgument(
        "Argument '", arg_def.name(),
        "' incorrectly specified in op definition: ", SummarizeOpDef(op_def));
=======
    return errors::InvalidArgument("Argument '", arg_def.name(),
                                   "' incorrectly specified in op definition: ",
                                   SummarizeOpDef(op_def));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
  return Status::OK();
}

<<<<<<< HEAD
Status NameRangesHelper(const AttrSlice& attrs,
=======
Status NameRangesHelper(const NodeDef& node_def,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                        const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
                        const OpDef& op_def, NameRangeMap* result) {
  int start = 0;
  int num;
  for (const auto& arg : args) {
<<<<<<< HEAD
    TF_RETURN_IF_ERROR(ComputeArgRange(attrs, arg, op_def, &num));
=======
    TF_RETURN_IF_ERROR(ComputeArgRange(node_def, arg, op_def, &num));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    (*result)[arg.name()] = std::make_pair(start, start + num);
    start += num;
  }
  return Status::OK();
}

}  // namespace

<<<<<<< HEAD
Status NameRangesForNode(const AttrSlice& attrs, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs) {
  if (inputs != nullptr) {
    TF_RETURN_IF_ERROR(
        NameRangesHelper(attrs, op_def.input_arg(), op_def, inputs));
  }
  if (outputs != nullptr) {
    return NameRangesHelper(attrs, op_def.output_arg(), op_def, outputs);
  }
  return Status::OK();
=======
Status NameRangesForNode(const NodeDef& node_def, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs) {
  TF_RETURN_IF_ERROR(
      NameRangesHelper(node_def, op_def.input_arg(), op_def, inputs));
  return NameRangesHelper(node_def, op_def.output_arg(), op_def, outputs);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

void AddDefaultsToNodeDef(const OpDef& op_def, NodeDef* node_def) {
  for (const auto& attr_def : op_def.attr()) {
    AttrSlice attrs(*node_def);
    if (attr_def.has_default_value() && !attrs.Find(attr_def.name())) {
      AddNodeAttr(attr_def.name(), attr_def.default_value(), node_def);
    }
  }
}

namespace {

<<<<<<< HEAD
using ::tensorflow::tstring;
using ::tensorflow::strings::Scanner;

bool IsValidNodeName(StringPiece sp) {
  Scanner scanner(sp);
  scanner.One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);

  while (true) {
    if (!scanner.GetResult())  // Some error in previous iteration.
      return false;
    if (scanner.empty())  // No error, but nothing left, good.
      return true;

    // Absorb another name/namespace, starting with a '>'
    scanner.One(Scanner::RANGLE)
        .One(Scanner::LETTER_DIGIT_DOT)
        .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
  }
}

bool IsValidDataInputName(StringPiece sp) {
  // Data inputs are op_name, op_name:0, or op_name:12345.
  Scanner scan(sp);
  scan.One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);

  while (true) {
    if (!scan.GetResult())  // Some error in previous iteration.
      return false;
    if (scan.empty())  // No error, but nothing left, good.
      return true;

    if (scan.Peek() == ':') {  // Absorb identifier after the colon
      scan.OneLiteral(":");
      if (scan.Peek() == '0') {
        scan.OneLiteral("0");  // :0
      } else {
        scan.Many(Scanner::DIGIT);  // :[1-9][0-9]*
      }
    } else {
      // Absorb another name/namespace, starting with a '>'
      scan.One(Scanner::RANGLE)
          .One(Scanner::LETTER_DIGIT_DOT)
          .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
    }
  }
}

bool IsValidControlInputName(StringPiece sp) {
  Scanner scan(sp);
  scan.OneLiteral("^")
      .One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);

  while (true) {
    if (!scan.GetResult())  // Some error in previous iteration.
      return false;
    if (scan.empty())  // No error, but nothing left, good.
      return true;

    // Absorb another name/namespace, starting with a '>'
    scan.One(Scanner::RANGLE)
        .One(Scanner::LETTER_DIGIT_DOT)
        .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
  }
}
=======
static RE2* valid_op_name_pattern = new RE2("[A-Za-z0-9.][A-Za-z0-9_.\\-/]*");
static RE2* valid_data_input_pattern =
    new RE2("[A-Za-z0-9.][A-Za-z0-9_.\\-/]*(\\:(0|([1-9][0-9]*)))?");
static RE2* valid_control_input_pattern =
    new RE2("\\^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

}  // namespace

Status ValidateOpInput(const string& input_name, bool* is_control_input) {
  *is_control_input = false;
<<<<<<< HEAD
  if (IsValidDataInputName(input_name)) {
    return Status::OK();
  } else if (IsValidControlInputName(input_name)) {
=======
  if (RE2::FullMatch(input_name, *valid_data_input_pattern)) {
    return Status::OK();
  } else if (RE2::FullMatch(input_name, *valid_control_input_pattern)) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    *is_control_input = true;
    return Status::OK();
  } else {
    return errors::InvalidArgument("Illegal op input name '", input_name, "'");
  }
}

<<<<<<< HEAD
Status ValidateNodeName(const string& node_name) {
  if (IsValidNodeName(node_name)) {
    return Status::OK();
  } else {
    return errors::InvalidArgument("Illegal op name '", node_name, "'");
=======
Status ValidateOpName(const string& op_name) {
  if (RE2::FullMatch(op_name, *valid_op_name_pattern)) {
    return Status::OK();
  } else {
    return errors::InvalidArgument("Illegal op name '", op_name, "'");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
}

Status ValidateExternalNodeDefSyntax(const NodeDef& node_def) {
<<<<<<< HEAD
  Status s = ValidateNodeName(node_def.name());
=======
  Status s = ValidateOpName(node_def.name());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!s.ok()) {
    return AttachDef(s, node_def);
  }
  bool in_control_inputs = false;
  for (const string& input_name : node_def.input()) {
    bool is_control_input;
    s = ValidateOpInput(input_name, &is_control_input);
    if (!s.ok()) {
      return AttachDef(s, node_def);
    }

    if (in_control_inputs && !is_control_input) {
      return AttachDef(errors::InvalidArgument(
                           "All control inputs must follow all data inputs"),
                       node_def);
    }
    in_control_inputs = is_control_input;
  }
  return Status::OK();
}

<<<<<<< HEAD
Status AttachDef(const Status& status, const NodeDef& node_def,
                 bool allow_multiple_formatted_node) {
  Status ret = status;
  string node_error;
  if (!allow_multiple_formatted_node &&
      status.error_message().find("{{node ") != string::npos) {
    node_error = node_def.name();
  } else {
    node_error = FormatNodeDefForError(node_def);
  }
  errors::AppendToMessage(&ret, strings::StrCat(" [[", node_error, "]]"));
  return ret;
}

void AddNodeAttr(StringPiece name, const AttrValue& value, NodeDef* node_def) {
  node_def->mutable_attr()->insert(
      AttrValueMap::value_type(string(name), value));
}

void AddNodeAttr(StringPiece name, AttrValue&& value, NodeDef* node_def) {
  (*node_def->mutable_attr())[string(name)] = std::move(value);
}

#define ADD_NODE_ATTR(T)                                           \
  void AddNodeAttr(StringPiece name, T value, NodeDef* node_def) { \
    AttrValue attr_value;                                          \
    SetAttrValue(value, &attr_value);                              \
    AddNodeAttr(name, attr_value, node_def);                       \
  }
ADD_NODE_ATTR(StringPiece)
ADD_NODE_ATTR(const char*)
ADD_NODE_ATTR(int32)
ADD_NODE_ATTR(int64)
ADD_NODE_ATTR(float)
ADD_NODE_ATTR(double)
ADD_NODE_ATTR(bool)
ADD_NODE_ATTR(DataType)
ADD_NODE_ATTR(const PartialTensorShape&)
ADD_NODE_ATTR(const Tensor&)
ADD_NODE_ATTR(const TensorProto&)
ADD_NODE_ATTR(const NameAttrList&)
ADD_NODE_ATTR(gtl::ArraySlice<StringPiece>)
ADD_NODE_ATTR(gtl::ArraySlice<const char*>)
ADD_NODE_ATTR(gtl::ArraySlice<string>)
ADD_NODE_ATTR(gtl::ArraySlice<int32>)
ADD_NODE_ATTR(gtl::ArraySlice<int64>)
ADD_NODE_ATTR(gtl::ArraySlice<float>)
ADD_NODE_ATTR(gtl::ArraySlice<bool>)
ADD_NODE_ATTR(const std::vector<bool>&)
ADD_NODE_ATTR(gtl::ArraySlice<DataType>)
ADD_NODE_ATTR(gtl::ArraySlice<TensorShape>)
ADD_NODE_ATTR(gtl::ArraySlice<PartialTensorShape>)
ADD_NODE_ATTR(gtl::ArraySlice<TensorShapeProto>)
ADD_NODE_ATTR(gtl::ArraySlice<Tensor>)
ADD_NODE_ATTR(gtl::ArraySlice<NameAttrList>)
#undef ADD_NODE_ATTR

void AddAttr(StringPiece name, const AttrValue& value, AttrValueMap* map) {
  map->insert(AttrValueMap::value_type(string(name), value));
}

#define ADD_ATTR(T)                                            \
  void AddAttr(StringPiece name, T value, AttrValueMap* map) { \
    AttrValue attr_value;                                      \
    SetAttrValue(value, &attr_value);                          \
    AddAttr(name, attr_value, map);                            \
  }
ADD_ATTR(bool)
#undef ADD_ATTR

Status AddPrefixAndSuffixToNode(StringPiece prefix, StringPiece suffix,
                                NodeDef* node_def, bool uniquify_frame_name) {
  node_def->set_name(strings::StrCat(prefix, node_def->name(), suffix));

  // Update frame name to avoid multiple LoopCond nodes in one frame.
  if (uniquify_frame_name &&
      (node_def->op() == "Enter" || node_def->op() == "RefEnter")) {
    string frame_name;
    TF_RETURN_IF_ERROR(GetNodeAttr(*node_def, "frame_name", &frame_name));
    AttrValue& attr = (*node_def->mutable_attr())["frame_name"];
    frame_name = strings::StrCat(prefix, frame_name, suffix);
    attr.set_s(frame_name);
  }

  // Update colocation constraints.
  constexpr char kClassAttr[] = "_class";
  auto class_attr = node_def->mutable_attr()->find(kClassAttr);
  if (class_attr != node_def->mutable_attr()->end()) {
    AttrValue new_value;
    new_value.mutable_list()->add_s(
        strings::StrCat(prefix, class_attr->second.s()));
    node_def->mutable_attr()->erase(kClassAttr);
    node_def->mutable_attr()->insert({kClassAttr, new_value});
  }

  return Status::OK();
}

=======
Status AttachDef(const Status& status, const NodeDef& node_def) {
  Status ret = status;
  errors::AppendToMessage(
      &ret, strings::StrCat(" [[Node: ", SummarizeNodeDef(node_def), "]]"));
  return ret;
}

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}  // namespace tensorflow
