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

#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace xla {
namespace tools {

namespace {

using tensorflow::StringPiece;
using tensorflow::gtl::optional;
using tensorflow::str_util::Split;
using tensorflow::str_util::SplitAndParseAsInts;
using tensorflow::strings::Printf;
using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;

const double kF16max = 65504;

// Parser for the HloModule::ToString() format text.
class HloParser {
 public:
  explicit HloParser(StringPiece str, const HloModuleConfig& config)
      : lexer_(str), config_(config) {}

  // Runs the parser. Returns false if an error occurred.
  bool Run();

  // Returns the parsed HloModule.
  std::unique_ptr<HloModule> ConsumeHloModule() { return std::move(module_); }

  // Returns the error information.
  string GetError() const { return tensorflow::str_util::Join(error_, "\n"); }

 private:
  // ParseXXX returns false if an error occurred.
  bool ParseHloModule();
  bool ParseComputations();
  bool ParseComputation();
  bool ParseInstructionList(HloComputation::Builder* builder,
                            string* root_name);
  bool ParseInstruction(HloComputation::Builder* builder, string* root_name);
  bool ParseControlPredecessors(HloInstruction* instruction);
  bool ParseLiteral(std::unique_ptr<Literal>* literal, const Shape& shape);
  bool ParseTupleLiteral(std::unique_ptr<Literal>* literal, const Shape& shape);
  bool ParseNonTupleLiteral(std::unique_ptr<Literal>* literal,
                            const Shape& shape);
  // Sets the sub-value of literal at the given index to the given value. The
  // literal's shape must have the default layout.
  bool SetValueInLiteral(int64 value, int64 linear_index, Literal* literal);
  bool SetValueInLiteral(double value, int64 linear_index, Literal* literal);
  bool SetValueInLiteral(bool value, int64 linear_index, Literal* literal);
  template <typename LiteralNativeT, typename ParsedElemT>
  bool SetValueInLiteralHelper(ParsedElemT value, int64 linear_index,
                               Literal* literal);

  bool ParseOperands(std::vector<HloInstruction*>* operands);
  // Fills parsed operands into 'operands' and expects a certain number of
  // operands.
  bool ParseOperands(std::vector<HloInstruction*>* operands,
                     const int expected_size);

  // Types of attributes.
  enum class AttrTy {
    kInt64,
    kHloComputation,
    kWindow,
    kConvolutionDimensionNumbers,
    kSharding,
    kInstructionList,
  };

  struct AttrConfig {
    bool required;     // whether it's required or optional
    AttrTy attr_type;  // what type it is
    void* result;      // where to store the parsed result.
  };

  // Parses attributes given names and configs of the attributes. Each parsed
  // result is passed back through the result pointer in corresponding
  // AttrConfig. Note that the result pointer must point to a optional<T> typed
  // variable which outlives this function. Returns false on error. You should
  // not use the any of the results if this function failed.
  //
  // Example usage:
  //
  //  std::unordered_map<string, AttrConfig> attrs;
  //  optional<int64> foo;
  //  attrs["foo"] = {/*required=*/false, AttrTy::kInt64, &foo};
  //  optional<Window> bar;
  //  attrs["bar"] = {/*required=*/true, AttrTy::kWindow, &bar};
  //  if (!ParseAttribute(attrs)) {
  //    return false; // Do not use 'foo' 'bar' if failed.
  //  }
  //  // Do something with 'bar'.
  //  if (foo) { // If attr foo is seen, do something with 'foo'. }
  //
  bool ParseAttributes(const std::unordered_map<string, AttrConfig>& attrs);

  // Parses a name and finds the corresponding hlo computation.
  bool ParseComputationName(HloComputation** value);
  // Parses a list of names and finds the corresponding hlo instructions.
  bool ParseInstructionNames(std::vector<HloInstruction*>* instructions);
  bool ParseWindow(Window* window);
  bool ParseConvolutionDimensionNumbers(ConvolutionDimensionNumbers* dnums);
  bool ParseSharding(OpSharding* sharding);
  bool ParseSingleSharding(OpSharding* sharding, bool lbrace_pre_lexed);

  // Parses a sub-attribute of the window attribute, e.g.,size=1x2x3.
  bool ParseDxD(const string& name, std::vector<int64>* result);
  // Parses window's pad sub-attriute, e.g., pad=0_0x3x3.
  bool ParseWindowPad(std::vector<std::vector<int64>>* pad);

  bool ParseParamList();
  bool ParseName(string* result);
  bool ParseAttributeName(string* result);
  bool ParseShape(Shape* result);
  bool ParseOpcode(HloOpcode* result);
  bool ParseInt64(int64* result);
  bool ParseDouble(double* result);
  bool ParseBool(bool* result);
  bool ParseToken(TokKind kind, const string& msg);

  // Logs the current parsing line and the given message. Always returns false.
  bool TokenError(StringPiece msg);

  // If the current token is 'kind', eats it (i.e. lexes the next token) and
  // returns true.
  bool EatIfPresent(TokKind kind);
  // Parses a shape, and returns true if the result is compatible with the given
  // shape.
  bool EatShapeAndCheckCompatible(const Shape& shape);

  // Adds the instruction to the pool. Returns false and emits an error if the
  // instruction already exists.
  bool AddInstruction(const string& name, HloInstruction* instruction);
  // Adds the computation to the pool. Returns false and emits an error if the
  // computation already exists.
  bool AddComputation(const string& name, HloComputation* computation);

  // The map from the instruction name to the instruction. This does not own the
  // instructions.
  std::unordered_map<string, HloInstruction*> instruction_pool_;
  std::unordered_map<string, HloComputation*> computation_pool_;

  HloLexer lexer_;
  std::unique_ptr<HloModule> module_;
  const HloModuleConfig config_;
  std::vector<string> error_;
};

bool HloParser::TokenError(StringPiece msg) {
  const string error =
      StrCat("was parsing \"", lexer_.GetCurrentLine(), "\"; token ",
             TokKindToString(lexer_.GetKind()), "; ", msg);
  VLOG(1) << "TokenError: " << error;
  error_.push_back(error);
  return false;
}

bool HloParser::Run() {
  lexer_.Lex();
  return ParseHloModule();
}

// ::= 'HloModule' name computations
bool HloParser::ParseHloModule() {
  if (lexer_.GetKind() != TokKind::kw_HloModule) {
    return TokenError("expects HloModule");
  }
  // Eat 'HloModule'
  lexer_.Lex();

  string name;
  if (!ParseName(&name)) {
    return false;
  }

  module_ = MakeUnique<HloModule>(name, config_);

  return ParseComputations();
}

// computations ::= (computation)+
bool HloParser::ParseComputations() {
  do {
    if (!ParseComputation()) {
      return false;
    }
  } while (lexer_.GetKind() != TokKind::kEof);
  return true;
}

// computation ::= ('ENTRY')? name param_list '->' shape instruction_list
bool HloParser::ParseComputation() {
  const bool is_entry_computation = EatIfPresent(TokKind::kw_ENTRY);
  string name;
  if (!ParseName(&name)) {
    return false;
  }
  auto builder = MakeUnique<HloComputation::Builder>(name);

  Shape shape;
  string root_name;
  if (!ParseParamList() || !ParseToken(TokKind::kArrow, "expects '->'") ||
      !ParseShape(&shape) || !ParseInstructionList(builder.get(), &root_name)) {
    return false;
  }

  HloInstruction* root =
      tensorflow::gtl::FindPtrOrNull(instruction_pool_, root_name);
  // This means some instruction was marked as ROOT but we didn't find it in the
  // pool, which should not happen.
  if (!root_name.empty() && root == nullptr) {
    LOG(FATAL) << "instruction " << root_name
               << " was marked as ROOT but the parser has not seen it before";
  }
  // Now root can be either an existing instruction or a nullptr. If it's a
  // nullptr, the implementation of Builder will set the last instruction as
  // root instruction.
  HloComputation* computation =
      is_entry_computation
          ? module_->AddEntryComputation(builder->Build(root))
          : module_->AddEmbeddedComputation(builder->Build(root));
  return AddComputation(name, computation);
}

// instruction_list ::= '{' instruction_list1 '}'
// instruction_list1 ::= (instruction)+
bool HloParser::ParseInstructionList(HloComputation::Builder* builder,
                                     string* root_name) {
  if (!ParseToken(TokKind::kLbrace,
                  "expects '{' at the beginning of instruction list.")) {
    return false;
  }
  do {
    if (!ParseInstruction(builder, root_name)) {
      return false;
    }
  } while (lexer_.GetKind() != TokKind::kRbrace);
  return ParseToken(TokKind::kRbrace,
                    "expects '}' at the end of instruction list.");
}

// instruction ::= ('ROOT')? name '=' shape opcode operands (attribute)*
bool HloParser::ParseInstruction(HloComputation::Builder* builder,
                                 string* root_name) {
  string name;
  Shape shape;
  HloOpcode opcode;
  std::vector<HloInstruction*> operands;
  bool is_root = EatIfPresent(TokKind::kw_ROOT);
  if (!ParseName(&name) ||
      !ParseToken(TokKind::kEqual, "expects '=' in instruction") ||
      !ParseShape(&shape) || !ParseOpcode(&opcode)) {
    return false;
  }
  if (is_root) {
    *root_name = name;
  }

  // Add optional attributes.
  std::unordered_map<string, AttrConfig> attrs;
  optional<OpSharding> sharding;
  attrs["sharding"] = {/*required=*/false, AttrTy::kSharding, &sharding};
  optional<std::vector<HloInstruction*>> predecessors;
  attrs["control-predecessors"] = {/*required=*/false, AttrTy::kInstructionList,
                                   &predecessors};

  HloInstruction* instruction;
  switch (opcode) {
    case HloOpcode::kParameter: {
      int64 parameter_number;
      if (!ParseToken(TokKind::kLparen,
                      "expects '(' before parameter number") ||
          !ParseInt64(&parameter_number) ||
          !ParseToken(TokKind::kRparen, "expects ')' after parameter number") ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateParameter(parameter_number, shape, name));
      break;
    }
    case HloOpcode::kConstant: {
      std::unique_ptr<Literal> literal;
      if (!ParseToken(TokKind::kLparen,
                      "expects '(' before constant literal") ||
          !ParseLiteral(&literal, shape) ||
          !ParseToken(TokKind::kRparen, "expects ')' after constant literal") ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
      break;
    }
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kReal:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSort:
    case HloOpcode::kTanh: {
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateUnary(shape, opcode, operands[0]));
      break;
    }
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kNe:
    case HloOpcode::kDot:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical: {
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateBinary(
          shape, opcode, operands[0], operands[1]));
      break;
    }
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect: {
      if (!ParseOperands(&operands, /*expected_size=*/3) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateTernary(
          shape, opcode, operands[0], operands[1], operands[2]));
      break;
    }
    // Other supported ops.
    case HloOpcode::kConvert: {
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateConvert(shape, operands[0]));
      break;
    }
    case HloOpcode::kCrossReplicaSum: {
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateCrossReplicaSum(shape, operands[0]));
      break;
    }
    case HloOpcode::kReshape: {
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateReshape(shape, operands[0]));
      break;
    }
    case HloOpcode::kTuple: {
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateTuple(operands));
      break;
    }
    case HloOpcode::kWhile: {
      optional<HloComputation*> condition;
      optional<HloComputation*> body;
      attrs["condition"] = {/*required=*/true, AttrTy::kHloComputation,
                            &condition};
      attrs["body"] = {/*required=*/true, AttrTy::kHloComputation, &body};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateWhile(
          shape, *condition, *body, /*init=*/operands[0]));
      break;
    }
    case HloOpcode::kRecv: {
      optional<int64> channel_id;
      attrs["channel_id"] = {/*required=*/true, AttrTy::kInt64, &channel_id};
      if (!ParseOperands(&operands, /*expected_size=*/0) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateRecv(shape, *channel_id));
      break;
    }
    case HloOpcode::kSend: {
      optional<int64> channel_id;
      attrs["channel_id"] = {/*required=*/true, AttrTy::kInt64, &channel_id};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateSend(operands[0], *channel_id));
      break;
    }
    case HloOpcode::kGetTupleElement: {
      optional<int64> index;
      attrs["index"] = {/*required=*/true, AttrTy::kInt64, &index};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateGetTupleElement(shape, operands[0], *index));
      break;
    }
    case HloOpcode::kCall: {
      optional<HloComputation*> to_apply;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &to_apply};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateCall(shape, operands, *to_apply));
      break;
    }
    case HloOpcode::kReduceWindow: {
      optional<HloComputation*> reduce_computation;
      optional<Window> window;
      attrs["window"] = {/*required=*/true, AttrTy::kWindow, &window};
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &reduce_computation};
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateReduceWindow(
          shape, /*operand=*/operands[0], /*init_value=*/operands[1], *window,
          *reduce_computation));
      break;
    }
    case HloOpcode::kConvolution: {
      optional<Window> window;
      optional<ConvolutionDimensionNumbers> dnums;
      attrs["window"] = {/*required=*/true, AttrTy::kWindow, &window};
      attrs["dim_labels"] = {/*required=*/true,
                             AttrTy::kConvolutionDimensionNumbers, &dnums};
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateConvolve(
          shape, /*lhs=*/operands[0], /*rhs=*/operands[1], *window, *dnums));
      break;
    }
    case HloOpcode::kBroadcast:
    case HloOpcode::kCustomCall:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kMap:
    case HloOpcode::kPad:
    case HloOpcode::kReduce:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kReverse:
    case HloOpcode::kRng:
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kFusion:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kTrace:
      return TokenError(StrCat("parsing not yet implemented for op: ",
                               HloOpcodeString(opcode)));
  }

  // Add common attrs (sharding, control predecessors) to the instruction, if
  // they were seen.
  if (sharding) {
    instruction->set_sharding(
        HloSharding::FromProto(sharding.value()).ValueOrDie());
  }
  if (predecessors) {
    for (auto* pre : *predecessors) {
      Status status = pre->AddControlDependencyTo(instruction);
      if (!status.ok()) {
        return TokenError(StrCat("error adding control dependency for: ", name,
                                 " status: ", status.ToString()));
      }
    }
  }
  return AddInstruction(name, instruction);
}

// ::= '{' (single_sharding | tuple_sharding) '}'
//
// tuple_sharding ::= single_sharding* (',' single_sharding)*
bool HloParser::ParseSharding(OpSharding* sharding) {
  // A single sharding starts with '{' and is not followed by '{'.
  // A tuple sharding starts with '{' and is followed by '{', or is '{''}' for
  // an empty tuple.
  if (!ParseToken(TokKind::kLbrace,
                  "expected '{' to start sharding attribute")) {
    return false;
  }

  if (lexer_.GetKind() != TokKind::kLbrace &&
      lexer_.GetKind() != TokKind::kRbrace) {
    return ParseSingleSharding(sharding, /*lbrace_pre_lexed=*/true);
  }

  // Tuple sharding.
  // Allow empty tuple shardings.
  if (lexer_.GetKind() != TokKind::kRbrace) {
    do {
      if (!ParseSingleSharding(sharding->add_tuple_shardings(),
                               /*lbrace_pre_lexed=*/false)) {
        return false;
      }
    } while (EatIfPresent(TokKind::kComma));
  }
  sharding->set_type(OpSharding::Type::OpSharding_Type_TUPLE);

  return ParseToken(TokKind::kRbrace, "expected '}' to end sharding attribute");
}

//  ::= '{' 'replicated'? 'maximal'? ('device=' int)? shape?
//          ('devices=' ('[' dims ']')* device_list)? '}'
// dims ::= int_list device_list ::= int_list
bool HloParser::ParseSingleSharding(OpSharding* sharding,
                                    bool lbrace_pre_lexed) {
  if (!lbrace_pre_lexed &&
      !ParseToken(TokKind::kLbrace,
                  "expected '{' to start sharding attribute")) {
    return false;
  }

  bool maximal = false;
  bool replicated = false;
  std::vector<int64> devices;
  std::vector<int64> tile_assignment_dimensions;
  Shape tile_shape;
  while (lexer_.GetKind() != TokKind::kRbrace) {
    switch (lexer_.GetKind()) {
      case TokKind::kw_maximal:
        maximal = true;
        lexer_.Lex();
        break;
      case TokKind::kw_replicated:
        replicated = true;
        lexer_.Lex();
        break;
      case TokKind::kAttributeName: {
        if (lexer_.GetStrVal() == "device") {
          if (lexer_.Lex() != TokKind::kInt) {
            return TokenError("device= attribute must be an integer");
          }
          devices = {lexer_.GetInt64Val()};
          lexer_.Lex();
        } else if (lexer_.GetStrVal() == "devices") {
          lexer_.Lex();
          if (!ParseToken(TokKind::kLsquare,
                          "expected '[' to start sharding devices shape")) {
            return false;
          }

          do {
            int64 dim;
            if (!ParseInt64(&dim)) {
              return false;
            }
            tile_assignment_dimensions.push_back(dim);
          } while (EatIfPresent(TokKind::kComma));

          if (!ParseToken(TokKind::kRsquare,
                          "expected ']' to start sharding devices shape")) {
            return false;
          }
          do {
            int64 device;
            if (!ParseInt64(&device)) {
              return false;
            }
            devices.push_back(device);
          } while (EatIfPresent(TokKind::kComma));
        } else {
          return TokenError(
              "unknown attribute in sharding: expected device= or devices=");
        }
        break;
      }
      case TokKind::kShape:
        tile_shape = lexer_.GetShapeVal();
        lexer_.Lex();
        break;
      case TokKind::kRbrace:
        break;
      default:
        return TokenError("unexpected token");
    }
  }

  if (replicated) {
    if (!devices.empty()) {
      return TokenError(
          "replicated shardings should not have any devices assigned");
    }
    if (!ShapeUtil::Equal(tile_shape, Shape())) {
      return TokenError(
          "replicated shardings should not have any tile shape set");
    }
    sharding->set_type(OpSharding::Type::OpSharding_Type_REPLICATED);
  } else if (maximal) {
    if (devices.size() != 1) {
      return TokenError(
          "maximal shardings should have exactly one device assigned");
    }
    if (!ShapeUtil::Equal(tile_shape, Shape())) {
      return TokenError("maximal shardings should not have any tile shape set");
    }
    sharding->set_type(OpSharding::Type::OpSharding_Type_MAXIMAL);
    sharding->add_tile_assignment_devices(devices[0]);
  } else {
    if (devices.size() <= 1) {
      return TokenError(
          "non-maximal shardings must have more than one device assigned");
    }
    if (ShapeUtil::Equal(tile_shape, Shape())) {
      return TokenError("non-maximal shardings should have a tile shape set");
    }
    if (tile_assignment_dimensions.empty()) {
      return TokenError(
          "non-maximal shardings must have a tile assignment list including "
          "dimensions");
    }
    sharding->set_type(OpSharding::Type::OpSharding_Type_OTHER);
    *sharding->mutable_tile_shape() = tile_shape;
    for (int64 dim : tile_assignment_dimensions) {
      sharding->add_tile_assignment_dimensions(dim);
    }
    for (int64 device : devices) {
      sharding->add_tile_assignment_devices(device);
    }
  }

  lexer_.Lex();
  return true;
}

// '{' name+ '}'
bool HloParser::ParseInstructionNames(
    std::vector<HloInstruction*>* instructions) {
  if (!ParseToken(TokKind::kLbrace,
                  "expects '{' at the beginning of instruction name list")) {
    return false;
  }
  do {
    string name;
    if (!ParseName(&name)) {
      return TokenError("expects a instruction name");
    }
    HloInstruction* instr =
        tensorflow::gtl::FindPtrOrNull(instruction_pool_, name);
    if (!instr) {
      return TokenError(
          Printf("instruction '%s' is not defined", name.c_str()));
    }
    instructions->push_back(instr);
  } while (EatIfPresent(TokKind::kComma));

  return ParseToken(TokKind::kRbrace,
                    "expects '}' at the end of control instructions");
}

bool HloParser::SetValueInLiteral(int64 value, int64 linear_index,
                                  Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case S8:
      return SetValueInLiteralHelper<int8>(value, linear_index, literal);
    case S16:
      return SetValueInLiteralHelper<int16>(value, linear_index, literal);
    case S32:
      return SetValueInLiteralHelper<int32>(value, linear_index, literal);
    case S64:
      return SetValueInLiteralHelper<int64>(value, linear_index, literal);
    case U8:
      return SetValueInLiteralHelper<uint8>(value, linear_index, literal);
    case U16:
      return SetValueInLiteralHelper<uint8>(value, linear_index, literal);
    case U32:
      return SetValueInLiteralHelper<uint32>(value, linear_index, literal);
    case U64:
      return SetValueInLiteralHelper<uint64>(value, linear_index, literal);
    default:
      LOG(FATAL) << "unknown integral primitive type "
                 << PrimitiveType_Name(shape.element_type());
  }
}

bool HloParser::SetValueInLiteral(double value, int64 linear_index,
                                  Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case F16:
      return SetValueInLiteralHelper<half>(value, linear_index, literal);
    case F32:
      return SetValueInLiteralHelper<float>(value, linear_index, literal);
    case F64:
      return SetValueInLiteralHelper<double>(value, linear_index, literal);
    default:
      LOG(FATAL) << "unknown floating point primitive type "
                 << PrimitiveType_Name(shape.element_type());
  }
}

bool HloParser::SetValueInLiteral(bool value, int64 linear_index,
                                  Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case PRED:
      return SetValueInLiteralHelper<bool>(value, linear_index, literal);
    default:
      LOG(FATAL) << PrimitiveType_Name(shape.element_type())
                 << " is not PRED type";
  }
}

template <typename LiteralNativeT, typename ParsedElemT>
bool HloParser::SetValueInLiteralHelper(ParsedElemT value, int64 linear_index,
                                        Literal* literal) {
  // Check that linear_index is in range.
  if (linear_index >= ShapeUtil::ElementsIn(literal->shape())) {
    return TokenError(
        StrCat("trys to set value ", value, " to a literal in shape ",
               ShapeUtil::HumanString(literal->shape()), " at linear index ",
               linear_index, ", but the index is out of range"));
  }

  if (std::isnan(value) ||
      (std::numeric_limits<ParsedElemT>::has_infinity &&
       (std::numeric_limits<ParsedElemT>::infinity() == value ||
        -std::numeric_limits<ParsedElemT>::infinity() == value))) {
    // Skip range checking for non-finite value.
  } else if (literal->shape().element_type() == F16) {
    if (value > kF16max || value < -kF16max) {
      return TokenError(StrCat(
          "value ", value, " is out of range for literal's primitive type ",
          PrimitiveType_Name(literal->shape().element_type())));
    }
  } else if (value > static_cast<ParsedElemT>(
                         std::numeric_limits<LiteralNativeT>::max()) ||
             value < static_cast<ParsedElemT>(
                         std::numeric_limits<LiteralNativeT>::lowest())) {
    // Value is out of range for LiteralNativeT.
    return TokenError(StrCat(
        "value ", value, " is out of range for literal's primitive type ",
        PrimitiveType_Name(literal->shape().element_type())));
  }

  literal->GetMutableArraySlice<LiteralNativeT>().at(linear_index) =
      static_cast<LiteralNativeT>(value);
  return true;
}

bool HloParser::EatShapeAndCheckCompatible(const Shape& shape) {
  Shape new_shape;
  if (!ParseShape(&new_shape)) {
    return TokenError(StrCat("expects shape ", ShapeUtil::HumanString(shape)));
  }
  if (!ShapeUtil::Compatible(shape, new_shape)) {
    return TokenError(StrCat(
        "expects shape ", ShapeUtil::HumanString(shape),
        ", but sees a different shape: ", ShapeUtil::HumanString(new_shape)));
  }
  return true;
}

// literal
//  ::= tuple
//  ::= non_tuple
bool HloParser::ParseLiteral(std::unique_ptr<Literal>* literal,
                             const Shape& shape) {
  return ShapeUtil::IsTuple(shape) ? ParseTupleLiteral(literal, shape)
                                   : ParseNonTupleLiteral(literal, shape);
}

// tuple
//  ::= shape '(' literal_list ')'
// literal_list
//  ::= /*empty*/
//  ::= literal (',' literal)*
bool HloParser::ParseTupleLiteral(std::unique_ptr<Literal>* literal,
                                  const Shape& shape) {
  if (!EatShapeAndCheckCompatible(shape)) {
    return TokenError(StrCat("expects tuple constant in shape ",
                             ShapeUtil::HumanString(shape)));
  }
  if (!ParseToken(TokKind::kLparen, "expects '(' in front of tuple elements")) {
    return false;
  }
  std::vector<std::unique_ptr<Literal>> elements(
      ShapeUtil::TupleElementCount(shape));

  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    // literal, (',' literal)*
    for (int i = 0; i < elements.size(); i++) {
      if (i > 0) {
        ParseToken(TokKind::kComma, "exepcts ',' to separate tuple elements");
      }
      if (!ParseLiteral(&elements[i],
                        ShapeUtil::GetTupleElementShape(shape, i))) {
        return TokenError(StrCat("expects the ", i, "th element"));
      }
    }
  }
  *literal = Literal::MakeTupleOwned(std::move(elements));
  return ParseToken(TokKind::kRparen,
                    StrCat("expects ')' at the end of the tuple with ",
                           ShapeUtil::TupleElementCount(shape), "elements"));
}

// non_tuple
//   ::= rank01
//   ::= rank2345
// rank2345 ::= shape nested_array
bool HloParser::ParseNonTupleLiteral(std::unique_ptr<Literal>* literal,
                                     const Shape& shape) {
  const int64 size = ShapeUtil::ElementsIn(shape);
  if (size == 0) {
    *literal = Literal::CreateFromShape(shape);
    return true;
  }

  const int64 rank = ShapeUtil::Rank(shape);
  if (rank > 1 && !EatShapeAndCheckCompatible(shape)) {
    return false;
  }

  // Create a literal with the given shape in default layout.
  *literal = Literal::CreateFromDimensions(shape.element_type(),
                                           AsInt64Slice(shape.dimensions()));
  int64 nest_level = 0;
  int64 linear_index = 0;
  // elems_seen_per_dim[i] is how many elements or sub-arrays we have seen for
  // the dimension i. For example, to parse f32[2,3] {{1, 2, 3}, {4, 5, 6}},
  // when we are parsing the 2nd '{' (right before '1'), we are seeing a
  // sub-array of the dimension 0, so elems_seen_per_dim[0]++. When we are at
  // the first '}' (right after '3'), it means the sub-array ends, and the
  // sub-array is supposed to contain exactly 3 elements, so check if
  // elems_seen_per_dim[1] is 3.
  std::vector<int64> elems_seen_per_dim(rank);
  auto get_index_str = [&elems_seen_per_dim](int dim) -> string {
    std::vector<int64> elems_seen_until_dim(elems_seen_per_dim.begin(),
                                            elems_seen_per_dim.begin() + dim);
    return StrCat("[",
                  tensorflow::str_util::Join(
                      elems_seen_until_dim, ",",
                      [](string* out, const int64& num_elems) {
                        tensorflow::strings::StrAppend(out, num_elems - 1);
                      }),
                  "]");
  };
  do {
    switch (lexer_.GetKind()) {
      default:
        return TokenError("unexpected token type in a literal");
      case TokKind::kLbrace: {
        nest_level++;
        if (nest_level > rank) {
          return TokenError(Printf(
              "expects nested array in rank %lld, but sees larger", rank));
        }
        if (nest_level > 1) {
          elems_seen_per_dim[nest_level - 2]++;
          if (elems_seen_per_dim[nest_level - 2] >
              shape.dimensions(nest_level - 2)) {
            return TokenError(Printf(
                "expects %lld elements in the %sth element, but sees more",
                shape.dimensions(nest_level - 2),
                get_index_str(nest_level - 2).c_str()));
          }
        }
        lexer_.Lex();
        break;
      }
      case TokKind::kRbrace: {
        nest_level--;
        if (elems_seen_per_dim[nest_level] != shape.dimensions(nest_level)) {
          return TokenError(Printf(
              "expects %lld elements in the %sth element, but sees %lld",
              shape.dimensions(nest_level), get_index_str(nest_level).c_str(),
              elems_seen_per_dim[nest_level]));
        }
        elems_seen_per_dim[nest_level] = 0;
        lexer_.Lex();
        break;
      }
      case TokKind::kComma:
      case TokKind::kComment:
        // Skip.
        lexer_.Lex();
        break;
      case TokKind::kw_true:
      case TokKind::kw_false:
      case TokKind::kInt:
      case TokKind::kDecimal:
      case TokKind::kw_nan:
      case TokKind::kw_inf:
      case TokKind::kNegInf: {
        if (rank > 0) {
          if (nest_level != rank) {
            return TokenError(
                Printf("expects nested array in rank %lld, but sees %lld", rank,
                       nest_level));
          }
          elems_seen_per_dim[rank - 1]++;
          if (elems_seen_per_dim[rank - 1] > shape.dimensions(rank - 1)) {
            return TokenError(
                Printf("expects %lld elements on the minor-most dimension, but "
                       "sees more",
                       shape.dimensions(rank - 1)));
          }
        }
        if (lexer_.GetKind() == TokKind::kw_true ||
            lexer_.GetKind() == TokKind::kw_false) {
          // TODO(congliu): bool type literals with rank >= 1 are actually
          // printed in a compact form instead of "true" or "false". Fix that.
          if (!SetValueInLiteral(lexer_.GetKind() == TokKind::kw_true,
                                 linear_index++, literal->get())) {
            return false;
          }
          lexer_.Lex();
        } else if (primitive_util::IsIntegralType(shape.element_type())) {
          int64 value;
          if (!ParseInt64(&value)) {
            return TokenError(StrCat("expects integer for primitive type: ",
                                     PrimitiveType_Name(shape.element_type())));
          }
          if (!SetValueInLiteral(value, linear_index++, literal->get())) {
            return false;
          }
        } else if (primitive_util::IsFloatingPointType(shape.element_type())) {
          double value;
          if (!ParseDouble(&value)) {
            return TokenError(
                StrCat("expect floating point value for primitive type: ",
                       PrimitiveType_Name(shape.element_type())));
          }
          if (!SetValueInLiteral(value, linear_index++, literal->get())) {
            return false;
          }
        } else {
          return TokenError(StrCat("unsupported premitive type ",
                                   PrimitiveType_Name(shape.element_type())));
        }
        break;
      }
    }  // end of switch
  } while (nest_level > 0);

  *literal = (*literal)->Relayout(shape.layout());
  return true;
}

// operands ::= '(' operands1 ')'
// operands1
//   ::= /*empty*/
//   ::= operand (, operand)*
// operand ::= shape name
bool HloParser::ParseOperands(std::vector<HloInstruction*>* operands) {
  if (!ParseToken(TokKind::kLparen,
                  "expects '(' at the beginning of operands")) {
    return false;
  }
  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    do {
      Shape shape;
      string name;
      if (!ParseShape(&shape) || !ParseName(&name)) {
        return false;
      }
      HloInstruction* instruction =
          tensorflow::gtl::FindPtrOrNull(instruction_pool_, name);
      if (!instruction) {
        return TokenError(StrCat("instruction does not exist: ", name));
      }
      operands->push_back(instruction);
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRparen, "expects ')' at the end of operands");
}

bool HloParser::ParseOperands(std::vector<HloInstruction*>* operands,
                              const int expected_size) {
  if (!ParseOperands(operands)) {
    return false;
  }
  if (expected_size != operands->size()) {
    return TokenError(StrCat("expects ", expected_size, " operands, but has ",
                             operands->size(), " operands"));
  }
  return true;
}

bool HloParser::ParseAttributes(
    const std::unordered_map<string, AttrConfig>& attrs) {
  std::unordered_set<string> seen_attrs;
  while (EatIfPresent(TokKind::kComma)) {
    string name;
    if (!ParseAttributeName(&name)) {
      return TokenError("error parsing attributes");
    }
    VLOG(1) << "Parsing attribute " << name;
    if (!seen_attrs.insert(name).second) {
      return TokenError(Printf("attribute %s already exists", name.c_str()));
    }
    auto attr_it = attrs.find(name);
    if (attr_it == attrs.end()) {
      return TokenError(Printf("unexpected attribute %s", name.c_str()));
    }
    AttrTy attr_type = attr_it->second.attr_type;
    void* attr_out_ptr = attr_it->second.result;
    bool success = [&] {
      switch (attr_type) {
        case AttrTy::kInt64: {
          int64 result;
          if (!ParseInt64(&result)) {
            return false;
          }
          static_cast<optional<int64>*>(attr_out_ptr)->emplace(result);
          return true;
        }
        case AttrTy::kHloComputation: {
          HloComputation* result;
          if (!ParseComputationName(&result)) {
            return false;
          }
          static_cast<optional<HloComputation*>*>(attr_out_ptr)
              ->emplace(result);
          return true;
        }
        case AttrTy::kWindow: {
          Window result;
          if (!ParseWindow(&result)) {
            return false;
          }
          static_cast<optional<Window>*>(attr_out_ptr)->emplace(result);
          return true;
        }
        case AttrTy::kConvolutionDimensionNumbers: {
          ConvolutionDimensionNumbers result;
          if (!ParseConvolutionDimensionNumbers(&result)) {
            return false;
          }
          static_cast<optional<ConvolutionDimensionNumbers>*>(attr_out_ptr)
              ->emplace(result);
          return true;
        }
        case AttrTy::kSharding: {
          OpSharding sharding;
          if (!ParseSharding(&sharding)) {
            return false;
          }
          static_cast<optional<OpSharding>*>(attr_out_ptr)->emplace(sharding);
          return true;
        }
        case AttrTy::kInstructionList: {
          std::vector<HloInstruction*> result;
          if (!ParseInstructionNames(&result)) {
            return false;
          }
          static_cast<optional<std::vector<HloInstruction*>>*>(attr_out_ptr)
              ->emplace(result);
          return true;
        }
      }
    }();
    if (!success) {
      return TokenError(Printf("error parsing attribute %s", name.c_str()));
    }
  }
  // Check that all required attrs were seen.
  for (const auto& attr_it : attrs) {
    if (attr_it.second.required &&
        seen_attrs.find(attr_it.first) == seen_attrs.end()) {
      return TokenError(Printf("attribute %s is expected but not seen",
                               attr_it.first.c_str()));
    }
  }
  return true;
}

bool HloParser::ParseComputationName(HloComputation** value) {
  string name;
  if (!ParseName(&name)) {
    return TokenError("expects computation name");
  }
  *value = tensorflow::gtl::FindPtrOrNull(computation_pool_, name);
  if (*value == nullptr) {
    return TokenError(StrCat("computation does not exist: ", name));
  }
  return true;
}

// ::= '{' size stride? pad? lhs_dilate? rhs_dilate? '}'
// The subattributes can appear in any order. 'size=' is required, others are
// optional.
bool HloParser::ParseWindow(Window* window) {
  if (!ParseToken(TokKind::kLbrace, "expected '{' to start window attribute")) {
    return false;
  }

  std::vector<int64> size;
  std::vector<int64> stride;
  std::vector<std::vector<int64>> pad;
  std::vector<int64> lhs_dilate;
  std::vector<int64> rhs_dilate;
  while (lexer_.GetKind() != TokKind::kRbrace) {
    string field_name;
    if (!ParseAttributeName(&field_name)) {
      return TokenError("expects sub-attributes in window");
    }
    bool ok = [&] {
      if (field_name == "size") {
        return ParseDxD("size", &size);
      }
      if (field_name == "stride") {
        return ParseDxD("stride", &stride);
      }
      if (field_name == "lhs_dilate") {
        return ParseDxD("lhs_dilate", &lhs_dilate);
      }
      if (field_name == "rhs_dilate") {
        return ParseDxD("rls_dilate", &rhs_dilate);
      }
      if (field_name == "pad") {
        return ParseWindowPad(&pad);
      }
      return TokenError(StrCat("unexpected attribute name: ", field_name));
    }();
    if (!ok) {
      return false;
    }
  }

  if (size.empty()) {
    return TokenError(
        "sub-attribute 'size=' is required in the window attribute");
  }
  if (!stride.empty() && stride.size() != size.size()) {
    return TokenError("expects 'stride=' has the same size as 'size='");
  }
  if (!lhs_dilate.empty() && lhs_dilate.size() != size.size()) {
    return TokenError("expects 'lhs_dilate=' has the same size as 'size='");
  }
  if (!rhs_dilate.empty() && rhs_dilate.size() != size.size()) {
    return TokenError("expects 'rhs_dilate=' has the same size as 'size='");
  }
  if (!pad.empty() && pad.size() != size.size()) {
    return TokenError("expects 'pad=' has the same size as 'size='");
  }

  for (int i = 0; i < size.size(); i++) {
    window->add_dimensions()->set_size(size[i]);
    if (!pad.empty()) {
      window->mutable_dimensions(i)->set_padding_low(pad[i][0]);
      window->mutable_dimensions(i)->set_padding_high(pad[i][1]);
    }
    // If some field is not present, it has the default value.
    window->mutable_dimensions(i)->set_stride(stride.empty() ? 1 : stride[i]);
    window->mutable_dimensions(i)->set_base_dilation(
        lhs_dilate.empty() ? 1 : lhs_dilate[i]);
    window->mutable_dimensions(i)->set_window_dilation(
        rhs_dilate.empty() ? 1 : rhs_dilate[i]);
  }
  return ParseToken(TokKind::kRbrace, "expected '}' to end window attribute");
}

// This is the inverse of HloInstruction::ConvolutionDimensionNumbersToString.
// The string looks like "dim_labels=0bf_0io->0bf".
bool HloParser::ParseConvolutionDimensionNumbers(
    ConvolutionDimensionNumbers* dnums) {
  if (lexer_.GetKind() != TokKind::kDimLabels) {
    return TokenError("expects dim labels pattern, e.g., 'bf0_0io->0bf'");
  }
  string str = lexer_.GetStrVal();

  // The str is expected to have 3 items, lhs, rhs, out, and it must looks like
  // lhs_rhs->out, that is, the first separator is "_" and the second is "->".
  // So we replace the "->" with "_" and then split on "_".
  str = tensorflow::str_util::StringReplace(str, /*oldsub=*/"->",
                                            /*newsub=*/"_",
                                            /*replace_all=*/false);
  std::vector<string> lhs_rhs_out = Split(str, "_");
  if (lhs_rhs_out.size() != 3) {
    LOG(FATAL) << "expects 3 items: lhs, rhs, and output dims, but sees "
               << str;
  }

  const int64 rank = lhs_rhs_out[0].length();
  if (rank != lhs_rhs_out[1].length() || rank != lhs_rhs_out[2].length()) {
    return TokenError(
        "convolution lhs, rhs, and output must have the same rank");
  }
  if (rank < 3) {
    return TokenError("convolution rank must >=3");
  }

  auto is_unique = [](string str) -> bool {
    std::sort(str.begin(), str.end());
    return std::unique(str.begin(), str.end()) == str.end();
  };

  // lhs
  {
    const string& lhs = lhs_rhs_out[0];
    if (!is_unique(lhs)) {
      return TokenError(
          StrCat("expects unique lhs dimension numbers, but sees ", lhs));
    }
    for (int i = 0; i < rank - 2; i++) {
      dnums->add_spatial_dimensions(-1);
    }
    for (int i = 0; i < rank; i++) {
      char c = lhs[i];
      if (c == 'b') {
        dnums->set_input_batch_dimension(i);
      } else if (c == 'f') {
        dnums->set_input_feature_dimension(i);
      } else if (c < '0' + rank && c >= '0') {
        dnums->set_spatial_dimensions(c - '0', i);
      } else {
        return TokenError(
            Printf("expects [0-%lldbf] in lhs dimension numbers", rank - 1));
      }
    }
  }
  // rhs
  {
    const string& rhs = lhs_rhs_out[1];
    if (!is_unique(rhs)) {
      return TokenError(
          StrCat("expects unique rhs dimension numbers, but sees ", rhs));
    }
    for (int i = 0; i < rank - 2; i++) {
      dnums->add_kernel_spatial_dimensions(-1);
    }
    for (int i = 0; i < rank; i++) {
      char c = rhs[i];
      if (c == 'i') {
        dnums->set_kernel_input_feature_dimension(i);
      } else if (c == 'o') {
        dnums->set_kernel_output_feature_dimension(i);
      } else if (c < '0' + rank && c >= '0') {
        dnums->set_kernel_spatial_dimensions(c - '0', i);
      } else {
        return TokenError(
            Printf("expects [0-%lldio] in rhs dimension numbers", rank - 1));
      }
    }
  }
  // output
  {
    const string& out = lhs_rhs_out[2];
    if (!is_unique(out)) {
      return TokenError(
          StrCat("expects unique output dimension numbers, but sees ", out));
    }
    for (int i = 0; i < rank; i++) {
      char c = out[i];
      if (c == 'b') {
        dnums->set_output_batch_dimension(i);
      } else if (c == 'f') {
        dnums->set_output_feature_dimension(i);
      } else if (c < '0' + rank && c >= '0') {
        if (dnums->spatial_dimensions(c - '0') != i) {
          return TokenError(
              "output spatial dimensions should be the same as input spatial "
              "dimensions");
        }
      } else {
        return TokenError(
            Printf("expects [0-%lldbf] in output dimension numbers", rank - 1));
      }
    }
  }

  lexer_.Lex();
  return true;
}

// param_list ::= '(' param_list1 ')'
// param_list1
//   ::= /*empty*/
//   ::= param (',' param)*
// param ::= name shape
bool HloParser::ParseParamList() {
  if (!ParseToken(TokKind::kLparen,
                  "expects '(' at the beginning of param list")) {
    return false;
  }

  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    do {
      Shape shape;
      if (!ParseToken(TokKind::kName, "expects name in parameter") ||
          !ParseShape(&shape)) {
        return false;
      }
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRparen, "expects ')' at the end of param list");
}

// shape ::= shape_val_
// shape ::= '(' tuple_elements ')'
// tuple_elements
//   ::= /*empty*/
//   ::= shape (',' shape)*
bool HloParser::ParseShape(Shape* result) {
  if (EatIfPresent(TokKind::kLparen)) {  // Tuple
    std::vector<Shape> shapes;
    if (lexer_.GetKind() == TokKind::kRparen) {
      /*empty*/
    } else {
      // shape (',' shape)*
      do {
        shapes.emplace_back();
        if (!ParseShape(&shapes.back())) {
          return false;
        }
      } while (EatIfPresent(TokKind::kComma));
    }
    *result = ShapeUtil::MakeTupleShape(shapes);
    return ParseToken(TokKind::kRparen, "expects ')' at the end of tuple.");
  }

  if (lexer_.GetKind() != TokKind::kShape) {
    return TokenError("expects shape");
  }
  *result = lexer_.GetShapeVal();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseName(string* result) {
  VLOG(1) << "ParseName";
  if (lexer_.GetKind() != TokKind::kName) {
    return TokenError("expects name");
  }
  *result = lexer_.GetStrVal();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseAttributeName(string* result) {
  if (lexer_.GetKind() != TokKind::kAttributeName) {
    return TokenError("expects attribute name");
  }
  *result = lexer_.GetStrVal();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseDxD(const string& name, std::vector<int64>* result) {
  if (!result->empty()) {
    return TokenError(
        Printf("sub-attribute '%s=' already exists", name.c_str()));
  }
  // 1D
  if (lexer_.GetKind() == TokKind::kInt) {
    int64 number;
    if (!ParseInt64(&number)) {
      return TokenError(Printf("expects sub-attribute '%s=i'", name.c_str()));
    }
    result->push_back(number);
    return true;
  }
  // 2D or higher.
  if (lexer_.GetKind() == TokKind::kDxD) {
    string str = lexer_.GetStrVal();
    if (!SplitAndParseAsInts(str, 'x', result)) {
      return TokenError(
          Printf("expects sub-attribute '%s=ixj...'", name.c_str()));
    }
    lexer_.Lex();
    return true;
  }
  return TokenError("expects token type kInt or kDxD");
}

bool HloParser::ParseWindowPad(std::vector<std::vector<int64>>* pad) {
  if (!pad->empty()) {
    return TokenError("sub-attribute 'pad=' already exists");
  }
  if (lexer_.GetKind() != TokKind::kWindowPad) {
    return TokenError("expects window pad pattern, e.g., '0_0x3_3'");
  }
  string str = lexer_.GetStrVal();
  std::vector<string> padding_str = Split(str, 'x');
  for (int i = 0; i < padding_str.size(); i++) {
    std::vector<int64> low_high;
    if (!SplitAndParseAsInts(padding_str[i], '_', &low_high) ||
        low_high.size() != 2) {
      return TokenError(
          "expects padding_low and padding_high separated by '_'");
    }
    pad->push_back(low_high);
  }
  lexer_.Lex();
  return true;
}

bool HloParser::ParseOpcode(HloOpcode* result) {
  VLOG(1) << "ParseOpcode";
  if (lexer_.GetKind() != TokKind::kOpcode) {
    return TokenError("expects opcode");
  }
  *result = lexer_.GetOpcodeVal();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseInt64(int64* result) {
  VLOG(1) << "ParseInt64";
  if (lexer_.GetKind() != TokKind::kInt) {
    return TokenError("expects integer");
  }
  *result = lexer_.GetInt64Val();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseDouble(double* result) {
  switch (lexer_.GetKind()) {
    case TokKind::kDecimal:
      *result = lexer_.GetDecimalVal();
      break;
    case TokKind::kInt:
      *result = static_cast<double>(lexer_.GetInt64Val());
      break;
    case TokKind::kw_nan:
      *result = std::numeric_limits<double>::quiet_NaN();
      break;
    case TokKind::kw_inf:
      *result = std::numeric_limits<double>::infinity();
      break;
    case TokKind::kNegInf:
      *result = -std::numeric_limits<double>::infinity();
      break;
    default:
      return TokenError("expects decimal or integer");
  }
  lexer_.Lex();
  return true;
}

bool HloParser::ParseBool(bool* result) {
  if (lexer_.GetKind() != TokKind::kw_true &&
      lexer_.GetKind() != TokKind::kw_false) {
    return TokenError("expects true or false");
  }
  *result = lexer_.GetKind() == TokKind::kw_true;
  lexer_.Lex();
  return true;
}

bool HloParser::ParseToken(TokKind kind, const string& msg) {
  VLOG(1) << "ParseToken " << TokKindToString(kind) << " " << msg;
  if (lexer_.GetKind() != kind) {
    return TokenError(msg);
  }
  lexer_.Lex();
  return true;
}

bool HloParser::EatIfPresent(TokKind kind) {
  if (lexer_.GetKind() != kind) {
    return false;
  }
  lexer_.Lex();
  return true;
}

bool HloParser::AddInstruction(const string& name,
                               HloInstruction* instruction) {
  auto result = instruction_pool_.insert({name, instruction});
  if (!result.second) {
    return TokenError(StrCat("instruction already exists: ", name));
  }
  return true;
}

bool HloParser::AddComputation(const string& name,
                               HloComputation* computation) {
  auto result = computation_pool_.insert({name, computation});
  if (!result.second) {
    return TokenError(StrCat("computation already exists: ", name));
  }
  return true;
}

}  // namespace

StatusOr<std::unique_ptr<HloModule>> Parse(StringPiece str,
                                           const HloModuleConfig& config) {
  HloParser parser(str, config);
  if (!parser.Run()) {
    return InvalidArgument("Syntax error: %s", parser.GetError().c_str());
  }
  return parser.ConsumeHloModule();
}

StatusOr<std::unique_ptr<HloModule>> Parse(StringPiece str) {
  HloModuleConfig config;
  return Parse(str, config);
}

}  // namespace tools
}  // namespace xla
