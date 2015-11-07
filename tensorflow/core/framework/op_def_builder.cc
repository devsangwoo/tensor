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

#include "tensorflow/core/framework/op_def_builder.h"

#include <limits>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/attr_value.pb.h"
=======
#include "tensorflow/core/framework/op_def_builder.h"

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
<<<<<<< HEAD
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

using ::tensorflow::strings::Scanner;
=======
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/regexp.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {

namespace {

<<<<<<< HEAD
string AttrError(StringPiece orig, const string& op_name) {
  return strings::StrCat(" from Attr(\"", orig, "\") for Op ", op_name);
}

bool ConsumeAttrName(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeListPrefix(StringPiece* sp) {
  return Scanner(*sp)
      .OneLiteral("list")
      .AnySpace()
      .OneLiteral("(")
      .AnySpace()
      .GetResult(sp);
}

bool ConsumeQuotedString(char quote_ch, StringPiece* sp, StringPiece* out) {
  const string quote_str(1, quote_ch);
  return Scanner(*sp)
      .OneLiteral(quote_str.c_str())
      .RestartCapture()
      .ScanEscapedUntil(quote_ch)
      .StopCapture()
      .OneLiteral(quote_str.c_str())
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeAttrType(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .Many(Scanner::LOWERLETTER_DIGIT)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeAttrNumber(StringPiece* sp, int64* out) {
  Scanner scan(*sp);
  StringPiece match;
  StringPiece remaining;

  scan.AnySpace().RestartCapture();
  if (scan.Peek() == '-') {
    scan.OneLiteral("-");
  }
  if (!scan.Many(Scanner::DIGIT)
           .StopCapture()
           .AnySpace()
           .GetResult(&remaining, &match)) {
    return false;
  }
  int64 value = 0;
  if (!strings::safe_strto64(match, &value)) {
    return false;
  }
  *out = value;
  *sp = remaining;
  return true;
=======
bool RE2Consume(StringPiece* sp, const char* pattern) {
  RegexpStringPiece base_sp = ToRegexpStringPiece(*sp);
  bool r = RE2::Consume(&base_sp, pattern);
  *sp = FromRegexpStringPiece(base_sp);
  return r;
}

bool RE2Consume(StringPiece* sp, const char* pattern, StringPiece* out) {
  RegexpStringPiece base_sp = ToRegexpStringPiece(*sp);
  RegexpStringPiece base_out;
  bool r = RE2::Consume(&base_sp, pattern, &base_out);
  *sp = FromRegexpStringPiece(base_sp);
  *out = FromRegexpStringPiece(base_out);
  return r;
}

bool RE2Consume(StringPiece* sp, const char* pattern, int64* out) {
  RegexpStringPiece base_sp = ToRegexpStringPiece(*sp);
  bool r = RE2::Consume(&base_sp, pattern, out);
  *sp = FromRegexpStringPiece(base_sp);
  return r;
}

string AttrError(StringPiece orig, const string& op_name) {
  return strings::StrCat(" from Attr(\"", orig, "\") for Op ", op_name);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

#define VERIFY(expr, ...)                                                 \
  do {                                                                    \
    if (!(expr)) {                                                        \
      errors->push_back(                                                  \
          strings::StrCat(__VA_ARGS__, AttrError(orig, op_def->name()))); \
      return;                                                             \
    }                                                                     \
  } while (false)

<<<<<<< HEAD
bool ConsumeCompoundAttrType(StringPiece* sp, StringPiece* out) {
  auto capture_begin = sp->begin();
  if (absl::ConsumePrefix(sp, "numbertype") ||
      absl::ConsumePrefix(sp, "numerictype") ||
      absl::ConsumePrefix(sp, "quantizedtype") ||
      absl::ConsumePrefix(sp, "realnumbertype") ||
      absl::ConsumePrefix(sp, "realnumberictype")) {
    *out = StringPiece(capture_begin, sp->begin() - capture_begin);
    return true;
  }
  return false;
}

bool ProcessCompoundType(const StringPiece type_string, AttrValue* allowed) {
  if (type_string == "numbertype" || type_string == "numerictype") {
    for (DataType dt : NumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (type_string == "quantizedtype") {
    for (DataType dt : QuantizedTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (type_string == "realnumbertype" ||
             type_string == "realnumerictype") {
    for (DataType dt : RealNumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else {
    return false;
  }
  return true;
}

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
void FinalizeAttr(StringPiece spec, OpDef* op_def,
                  std::vector<string>* errors) {
  OpDef::AttrDef* attr = op_def->add_attr();
  StringPiece orig(spec);

  // Parse "<name>:" at the beginning.
  StringPiece tmp_name;
<<<<<<< HEAD
  VERIFY(ConsumeAttrName(&spec, &tmp_name), "Trouble parsing '<name>:'");
  attr->set_name(tmp_name.data(), tmp_name.size());

  // Read "<type>" or "list(<type>)".
  bool is_list = ConsumeListPrefix(&spec);
  string type;
  StringPiece type_string;  // Used if type == "type"
  if (absl::ConsumePrefix(&spec, "string")) {
    type = "string";
  } else if (absl::ConsumePrefix(&spec, "int")) {
    type = "int";
  } else if (absl::ConsumePrefix(&spec, "float")) {
    type = "float";
  } else if (absl::ConsumePrefix(&spec, "bool")) {
    type = "bool";
  } else if (absl::ConsumePrefix(&spec, "type")) {
    type = "type";
  } else if (absl::ConsumePrefix(&spec, "shape")) {
    type = "shape";
  } else if (absl::ConsumePrefix(&spec, "tensor")) {
    type = "tensor";
  } else if (absl::ConsumePrefix(&spec, "func")) {
    type = "func";
  } else if (ConsumeCompoundAttrType(&spec, &type_string)) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    VERIFY(ProcessCompoundType(type_string, allowed),
           "Expected to see a compound type, saw: ", type_string);
  } else if (absl::ConsumePrefix(&spec, "{")) {
    // e.g. "{ int32, float, bool }" or "{ \"foo\", \"bar\" }"
    AttrValue* allowed = attr->mutable_allowed_values();
    str_util::RemoveLeadingWhitespace(&spec);
    if (absl::StartsWith(spec, "\"") || absl::StartsWith(spec, "'")) {
      type = "string";  // "{ \"foo\", \"bar\" }" or "{ 'foo', 'bar' }"
      while (true) {
        StringPiece escaped_string;
        VERIFY(ConsumeQuotedString('"', &spec, &escaped_string) ||
                   ConsumeQuotedString('\'', &spec, &escaped_string),
               "Trouble parsing allowed string at '", spec, "'");
        string unescaped;
        string error;
        VERIFY(absl::CUnescape(escaped_string, &unescaped, &error),
               "Trouble unescaping \"", escaped_string,
               "\", got error: ", error);
        allowed->mutable_list()->add_s(unescaped);
        if (absl::ConsumePrefix(&spec, ",")) {
          str_util::RemoveLeadingWhitespace(&spec);
          if (absl::ConsumePrefix(&spec, "}"))
            break;  // Allow ending with ", }".
        } else {
          VERIFY(absl::ConsumePrefix(&spec, "}"),
=======
  VERIFY(RE2Consume(&spec, "([a-zA-Z][a-zA-Z0-9_]*)\\s*:\\s*", &tmp_name),
         "Trouble parsing '<name>:'");
  attr->set_name(tmp_name.data(), tmp_name.size());

  // Read "<type>" or "list(<type>)".
  bool is_list = RE2Consume(&spec, "list\\s*\\(\\s*");
  string type;
  if (spec.Consume("string")) {
    type = "string";
  } else if (spec.Consume("int")) {
    type = "int";
  } else if (spec.Consume("float")) {
    type = "float";
  } else if (spec.Consume("bool")) {
    type = "bool";
  } else if (spec.Consume("type")) {
    type = "type";
  } else if (spec.Consume("shape")) {
    type = "shape";
  } else if (spec.Consume("tensor")) {
    type = "tensor";
  } else if (spec.Consume("func")) {
    type = "func";
  } else if (spec.Consume("numbertype") || spec.Consume("numerictype")) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    for (DataType dt : NumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (spec.Consume("quantizedtype")) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    for (DataType dt : QuantizedTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (spec.Consume("realnumbertype") ||
             spec.Consume("realnumerictype")) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    for (DataType dt : RealNumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (spec.Consume("{")) {
    // e.g. "{ int32, float, bool }" or "{ \"foo\", \"bar\" }"
    RE2Consume(&spec, "\\s*");
    AttrValue* allowed = attr->mutable_allowed_values();
    if (spec.starts_with("\"") || spec.starts_with("'")) {
      type = "string";  // "{ \"foo\", \"bar\" }" or "{ 'foo', 'bar' }"
      while (true) {
        StringPiece escaped_string;
        VERIFY((RE2Consume(&spec, R"xx("((?:[^"\\]|\\.)*)"\s*)xx",
                           &escaped_string) ||
                RE2Consume(&spec, R"xx('((?:[^'\\]|\\.)*)'\s*)xx",
                           &escaped_string)),
               "Trouble parsing allowed string at '", spec, "'");
        string unescaped;
        string error;
        VERIFY(str_util::CUnescape(escaped_string, &unescaped, &error),
               "Trouble unescaping \"", escaped_string, "\", got error: ",
               error);
        allowed->mutable_list()->add_s(unescaped);
        if (spec.Consume(",")) {
          RE2Consume(&spec, "\\s*");
          if (spec.Consume("}")) break;  // Allow ending with ", }".
        } else {
          VERIFY(spec.Consume("}"),
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                 "Expected , or } after strings in list, not: '", spec, "'");
          break;
        }
      }
<<<<<<< HEAD
    } else {  // "{ bool, numbertype, string }"
      type = "type";
      while (true) {
        VERIFY(ConsumeAttrType(&spec, &type_string),
               "Trouble parsing type string at '", spec, "'");
        if (ProcessCompoundType(type_string, allowed)) {
          // Processed a compound type.
        } else {
          DataType dt;
          VERIFY(DataTypeFromString(type_string, &dt),
                 "Unrecognized type string '", type_string, "'");
          allowed->mutable_list()->add_type(dt);
        }
        if (absl::ConsumePrefix(&spec, ",")) {
          str_util::RemoveLeadingWhitespace(&spec);
          if (absl::ConsumePrefix(&spec, "}"))
            break;  // Allow ending with ", }".
        } else {
          VERIFY(absl::ConsumePrefix(&spec, "}"),
=======
    } else {  // "{ int32, float, bool }"
      type = "type";
      while (true) {
        StringPiece type_string;
        VERIFY(RE2Consume(&spec, "([a-z0-9]+)\\s*", &type_string),
               "Trouble parsing type string at '", spec, "'");
        DataType dt;
        VERIFY(DataTypeFromString(type_string, &dt),
               "Unrecognized type string '", type_string, "'");
        allowed->mutable_list()->add_type(dt);
        if (spec.Consume(",")) {
          RE2Consume(&spec, "\\s*");
          if (spec.Consume("}")) break;  // Allow ending with ", }".
        } else {
          VERIFY(spec.Consume("}"),
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                 "Expected , or } after types in list, not: '", spec, "'");
          break;
        }
      }
    }
<<<<<<< HEAD
  } else {  // if spec.Consume("{")
    VERIFY(false, "Trouble parsing type string at '", spec, "'");
  }
  str_util::RemoveLeadingWhitespace(&spec);

  // Write the type into *attr.
  if (is_list) {
    VERIFY(absl::ConsumePrefix(&spec, ")"),
           "Expected ) to close 'list(', not: '", spec, "'");
    str_util::RemoveLeadingWhitespace(&spec);
=======
  } else {
    VERIFY(false, "Trouble parsing type string at '", spec, "'");
  }
  RE2Consume(&spec, "\\s*");

  // Write the type into *attr.
  if (is_list) {
    VERIFY(spec.Consume(")"), "Expected ) to close 'list(', not: '", spec, "'");
    RE2Consume(&spec, "\\s*");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    attr->set_type(strings::StrCat("list(", type, ")"));
  } else {
    attr->set_type(type);
  }

  // Read optional minimum constraint at the end.
<<<<<<< HEAD
  if ((is_list || type == "int") && absl::ConsumePrefix(&spec, ">=")) {
    int64 min_limit = -999;
    VERIFY(ConsumeAttrNumber(&spec, &min_limit),
=======
  if ((is_list || type == "int") && spec.Consume(">=")) {
    int64 min_limit = -999;
    VERIFY(RE2Consume(&spec, "\\s*(-?\\d+)\\s*", &min_limit),
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
           "Could not parse integer lower limit after '>=', found '", spec,
           "' instead");
    attr->set_has_minimum(true);
    attr->set_minimum(min_limit);
  }

  // Parse default value, if present.
<<<<<<< HEAD
  if (absl::ConsumePrefix(&spec, "=")) {
    str_util::RemoveLeadingWhitespace(&spec);
=======
  if (spec.Consume("=")) {
    RE2Consume(&spec, "\\s*");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    VERIFY(ParseAttrValue(attr->type(), spec, attr->mutable_default_value()),
           "Could not parse default value '", spec, "'");
  } else {
    VERIFY(spec.empty(), "Extra '", spec, "' unparsed at the end");
  }
}

#undef VERIFY

string InOutError(bool is_output, StringPiece orig, const string& op_name) {
  return strings::StrCat(" from ", is_output ? "Output" : "Input", "(\"", orig,
                         "\") for Op ", op_name);
}

<<<<<<< HEAD
bool ConsumeInOutName(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .One(Scanner::LOWERLETTER)
      .Any(Scanner::LOWERLETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeInOutRefOpen(StringPiece* sp) {
  return Scanner(*sp)
      .OneLiteral("Ref")
      .AnySpace()
      .OneLiteral("(")
      .AnySpace()
      .GetResult(sp);
}

bool ConsumeInOutRefClose(StringPiece* sp) {
  return Scanner(*sp).OneLiteral(")").AnySpace().GetResult(sp);
}

bool ConsumeInOutNameOrType(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeInOutTimesType(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .OneLiteral("*")
      .AnySpace()
      .RestartCapture()
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeControlOutName(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .GetResult(sp, out);
}

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#define VERIFY(expr, ...)                                             \
  do {                                                                \
    if (!(expr)) {                                                    \
      errors->push_back(strings::StrCat(                              \
          __VA_ARGS__, InOutError(is_output, orig, op_def->name()))); \
      return;                                                         \
    }                                                                 \
  } while (false)

void FinalizeInputOrOutput(StringPiece spec, bool is_output, OpDef* op_def,
                           std::vector<string>* errors) {
  OpDef::ArgDef* arg =
      is_output ? op_def->add_output_arg() : op_def->add_input_arg();

  StringPiece orig(spec);

  // Parse "<name>:" at the beginning.
  StringPiece tmp_name;
<<<<<<< HEAD
  VERIFY(ConsumeInOutName(&spec, &tmp_name), "Trouble parsing 'name:'");
  arg->set_name(tmp_name.data(), tmp_name.size());

  // Detect "Ref(...)".
  if (ConsumeInOutRefOpen(&spec)) {
=======
  VERIFY(RE2Consume(&spec, "([a-z][a-z0-9_]*)\\s*:\\s*", &tmp_name),
         "Trouble parsing 'name:'");
  arg->set_name(tmp_name.data(), tmp_name.size());

  // Detect "Ref(...)".
  if (RE2Consume(&spec, "Ref\\s*\\(\\s*")) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    arg->set_is_ref(true);
  }

  {  // Parse "<name|type>" or "<name>*<name|type>".
    StringPiece first, second, type_or_attr;
<<<<<<< HEAD
    VERIFY(ConsumeInOutNameOrType(&spec, &first),
           "Trouble parsing either a type or an attr name at '", spec, "'");
    if (ConsumeInOutTimesType(&spec, &second)) {
=======
    VERIFY(RE2Consume(&spec, "([a-zA-Z][a-zA-Z0-9_]*)\\s*", &first),
           "Trouble parsing either a type or an attr name at '", spec, "'");
    if (RE2Consume(&spec, "[*]\\s*([a-zA-Z][a-zA-Z0-9_]*)\\s*", &second)) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      arg->set_number_attr(first.data(), first.size());
      type_or_attr = second;
    } else {
      type_or_attr = first;
    }
    DataType dt;
    if (DataTypeFromString(type_or_attr, &dt)) {
      arg->set_type(dt);
    } else {
      const OpDef::AttrDef* attr = FindAttr(type_or_attr, *op_def);
      VERIFY(attr != nullptr, "Reference to unknown attr '", type_or_attr, "'");
      if (attr->type() == "type") {
        arg->set_type_attr(type_or_attr.data(), type_or_attr.size());
      } else {
        VERIFY(attr->type() == "list(type)", "Reference to attr '",
               type_or_attr, "' with type ", attr->type(),
               " that isn't type or list(type)");
        arg->set_type_list_attr(type_or_attr.data(), type_or_attr.size());
      }
    }
  }

  // Closing ) for Ref(.
  if (arg->is_ref()) {
<<<<<<< HEAD
    VERIFY(ConsumeInOutRefClose(&spec),
=======
    VERIFY(RE2Consume(&spec, "\\)\\s*"),
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
           "Did not find closing ')' for 'Ref(', instead found: '", spec, "'");
  }

  // Should not have anything else.
  VERIFY(spec.empty(), "Extra '", spec, "' unparsed at the end");

  // Int attrs that are the length of an input or output get a default
  // minimum of 1.
  if (!arg->number_attr().empty()) {
    OpDef::AttrDef* attr = FindAttrMutable(arg->number_attr(), op_def);
    if (attr != nullptr && !attr->has_minimum()) {
      attr->set_has_minimum(true);
      attr->set_minimum(1);
    }
  } else if (!arg->type_list_attr().empty()) {
    // If an input or output has type specified by a list(type) attr,
    // it gets a default minimum of 1 as well.
    OpDef::AttrDef* attr = FindAttrMutable(arg->type_list_attr(), op_def);
    if (attr != nullptr && attr->type() == "list(type)" &&
        !attr->has_minimum()) {
      attr->set_has_minimum(true);
      attr->set_minimum(1);
    }
  }
<<<<<<< HEAD

  // If the arg's dtype is resource we should mark the op as stateful as it
  // likely touches a resource manager. This deliberately doesn't cover inputs /
  // outputs which resolve to resource via Attrs as those mostly operate on
  // resource handles as an opaque type (as opposed to ops which explicitly take
  // / produce resources).
  if (arg->type() == DT_RESOURCE) {
    op_def->set_is_stateful(true);
  }
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

#undef VERIFY

<<<<<<< HEAD
string ControlOutError(StringPiece orig, const string& op_name) {
  return strings::StrCat(" from ControlOutput(\"", orig, "\") for Op ",
                         op_name);
}

void FinalizeControlOutput(StringPiece name, OpDef* op_def,
                           std::vector<string>* errors) {
  StringPiece orig(name);

  // Parse control output name.
  StringPiece tmp_name;
  if (!ConsumeControlOutName(&orig, &tmp_name)) {
    errors->push_back(strings::StrCat("Trouble parsing 'name:'",
                                      ControlOutError(orig, op_def->name())));
  }

  *op_def->add_control_output() = string(tmp_name.data(), tmp_name.size());
}

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
int num_leading_spaces(StringPiece s) {
  size_t i = 0;
  while (i < s.size() && s[i] == ' ') {
    ++i;
  }
  return i;
}

<<<<<<< HEAD
bool ConsumeDocNameColon(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool IsDocNameColon(StringPiece s) {
  return ConsumeDocNameColon(&s, nullptr /* out */);
}

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
void FinalizeDoc(const string& text, OpDef* op_def,
                 std::vector<string>* errors) {
  std::vector<string> lines = str_util::Split(text, '\n');

  // Remove trailing spaces.
  for (string& line : lines) {
<<<<<<< HEAD
    absl::StripTrailingAsciiWhitespace(&line);
=======
    str_util::StripTrailingWhitespace(&line);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  // First non-blank line -> summary.
  int l = 0;
  while (static_cast<size_t>(l) < lines.size() && lines[l].empty()) ++l;
  if (static_cast<size_t>(l) < lines.size()) {
    op_def->set_summary(lines[l]);
    ++l;
  }
  while (static_cast<size_t>(l) < lines.size() && lines[l].empty()) ++l;

  // Lines until we see name: -> description.
  int start_l = l;
<<<<<<< HEAD
  while (static_cast<size_t>(l) < lines.size() && !IsDocNameColon(lines[l])) {
=======
  while (static_cast<size_t>(l) < lines.size() &&
         !RE2::PartialMatch(lines[l], "^[a-zA-Z][a-zA-Z0-9_]*\\s*:")) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    ++l;
  }
  int end_l = l;
  // Trim trailing blank lines from the description.
  while (start_l < end_l && lines[end_l - 1].empty()) --end_l;
<<<<<<< HEAD
  string desc = absl::StrJoin(
=======
  string desc = str_util::Join(
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      gtl::ArraySlice<string>(lines.data() + start_l, end_l - start_l), "\n");
  if (!desc.empty()) op_def->set_description(desc);

  // name: description
  //   possibly continued on the next line
  //   if so, we remove the minimum indent
  StringPiece name;
  std::vector<StringPiece> description;
  while (static_cast<size_t>(l) < lines.size()) {
    description.clear();
    description.push_back(lines[l]);
<<<<<<< HEAD
    ConsumeDocNameColon(&description.back(), &name);
    ++l;
    while (static_cast<size_t>(l) < lines.size() && !IsDocNameColon(lines[l])) {
=======
    RE2Consume(&description.back(), "([a-zA-Z][a-zA-Z0-9_]*)\\s*:\\s*", &name);
    ++l;
    while (static_cast<size_t>(l) < lines.size() &&
           !RE2::PartialMatch(lines[l], "^[a-zA-Z][a-zA-Z0-9_]*\\s*:")) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      description.push_back(lines[l]);
      ++l;
    }
    // Remove any trailing blank lines.
    while (!description.empty() && description.back().empty()) {
      description.pop_back();
    }
    // Compute the minimum indent of all lines after the first.
    int min_indent = -1;
    for (size_t i = 1; i < description.size(); ++i) {
      if (!description[i].empty()) {
        int indent = num_leading_spaces(description[i]);
        if (min_indent < 0 || indent < min_indent) min_indent = indent;
      }
    }
    // Remove min_indent spaces from all lines after the first.
    for (size_t i = 1; i < description.size(); ++i) {
      if (!description[i].empty()) description[i].remove_prefix(min_indent);
    }
    // Concatenate lines into a single string.
<<<<<<< HEAD
    const string complete(absl::StrJoin(description, "\n"));
=======
    const string complete(str_util::Join(description, "\n"));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    // Find name.
    bool found = false;
    for (int i = 0; !found && i < op_def->input_arg_size(); ++i) {
      if (op_def->input_arg(i).name() == name) {
        op_def->mutable_input_arg(i)->set_description(complete);
        found = true;
      }
    }
    for (int i = 0; !found && i < op_def->output_arg_size(); ++i) {
      if (op_def->output_arg(i).name() == name) {
        op_def->mutable_output_arg(i)->set_description(complete);
        found = true;
      }
    }
    for (int i = 0; !found && i < op_def->attr_size(); ++i) {
      if (op_def->attr(i).name() == name) {
        op_def->mutable_attr(i)->set_description(complete);
        found = true;
      }
    }
    if (!found) {
      errors->push_back(
          strings::StrCat("No matching input/output/attr for name '", name,
                          "' from Doc() for Op ", op_def->name()));
      return;
    }
  }
}

}  // namespace

<<<<<<< HEAD
OpDefBuilder::OpDefBuilder(string op_name) {
  op_def()->set_name(std::move(op_name));
}

OpDefBuilder& OpDefBuilder::Attr(string spec) {
  attrs_.push_back(std::move(spec));
  return *this;
}

OpDefBuilder& OpDefBuilder::Input(string spec) {
  inputs_.push_back(std::move(spec));
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(string spec) {
  outputs_.push_back(std::move(spec));
  return *this;
}

OpDefBuilder& OpDefBuilder::ControlOutput(string name) {
  control_outputs_.push_back(std::move(name));
  return *this;
}

#ifndef TF_LEAN_BINARY
OpDefBuilder& OpDefBuilder::Doc(string text) {
  if (!doc_.empty()) {
    errors_.push_back(
        strings::StrCat("Extra call to Doc() for Op ", op_def()->name()));
  } else {
    doc_ = std::move(text);
  }
  return *this;
}
#endif

OpDefBuilder& OpDefBuilder::SetIsCommutative() {
  op_def()->set_is_commutative(true);
=======
OpDefBuilder::OpDefBuilder(StringPiece op_name) {
  op_def_.set_name(op_name.ToString());  // NOLINT
}

OpDefBuilder& OpDefBuilder::Attr(StringPiece spec) {
  attrs_.emplace_back(spec.data(), spec.size());
  return *this;
}

OpDefBuilder& OpDefBuilder::Input(StringPiece spec) {
  inputs_.emplace_back(spec.data(), spec.size());
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(StringPiece spec) {
  outputs_.emplace_back(spec.data(), spec.size());
  return *this;
}

OpDefBuilder& OpDefBuilder::Doc(StringPiece text) {
  if (!doc_.empty()) {
    errors_.push_back(
        strings::StrCat("Extra call to Doc() for Op ", op_def_.name()));
  } else {
    doc_.assign(text.data(), text.size());
  }
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsCommutative() {
  op_def_.set_is_commutative(true);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsAggregate() {
<<<<<<< HEAD
  op_def()->set_is_aggregate(true);
=======
  op_def_.set_is_aggregate(true);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsStateful() {
<<<<<<< HEAD
  op_def()->set_is_stateful(true);
=======
  op_def_.set_is_stateful(true);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return *this;
}

OpDefBuilder& OpDefBuilder::SetAllowsUninitializedInput() {
<<<<<<< HEAD
  op_def()->set_allows_uninitialized_input(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::Deprecated(int version, string explanation) {
  if (op_def()->has_deprecation()) {
    errors_.push_back(
        strings::StrCat("Deprecated called twice for Op ", op_def()->name()));
  } else {
    OpDeprecation* deprecation = op_def()->mutable_deprecation();
    deprecation->set_version(version);
    deprecation->set_explanation(std::move(explanation));
  }
  return *this;
}

OpDefBuilder& OpDefBuilder::SetShapeFn(OpShapeInferenceFn fn) {
  if (op_reg_data_.shape_inference_fn != nullptr) {
    errors_.push_back(
        strings::StrCat("SetShapeFn called twice for Op ", op_def()->name()));
  } else {
    op_reg_data_.shape_inference_fn = OpShapeInferenceFn(fn);
  }
  return *this;
}

Status OpDefBuilder::Finalize(OpRegistrationData* op_reg_data) const {
  std::vector<string> errors = errors_;
  *op_reg_data = op_reg_data_;

  OpDef* op_def = &op_reg_data->op_def;
=======
  op_def_.set_allows_uninitialized_input(true);
  return *this;
}

Status OpDefBuilder::Finalize(OpDef* op_def) const {
  std::vector<string> errors = errors_;
  *op_def = op_def_;

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  for (StringPiece attr : attrs_) {
    FinalizeAttr(attr, op_def, &errors);
  }
  for (StringPiece input : inputs_) {
    FinalizeInputOrOutput(input, false, op_def, &errors);
  }
  for (StringPiece output : outputs_) {
    FinalizeInputOrOutput(output, true, op_def, &errors);
  }
<<<<<<< HEAD
  for (StringPiece control_output : control_outputs_) {
    FinalizeControlOutput(control_output, op_def, &errors);
  }
  FinalizeDoc(doc_, op_def, &errors);

  if (errors.empty()) return Status::OK();
  return errors::InvalidArgument(absl::StrJoin(errors, "\n"));
=======
  FinalizeDoc(doc_, op_def, &errors);

  if (errors.empty()) return Status::OK();
  return errors::InvalidArgument(str_util::Join(errors, "\n"));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

}  // namespace tensorflow
