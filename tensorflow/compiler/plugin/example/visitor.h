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

#ifndef TENSORFLOW_COMPILER_EXAMPLE_VISITOR_H_
#define TENSORFLOW_COMPILER_EXAMPLE_VISITOR_H_

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"

namespace xla {
namespace exampleplugin {

class ExampleVisitor : public DfsHloVisitor {
public:
  ExampleVisitor();

  Status HandleElementwiseUnary(HloInstruction* hlo, HloOpcode opcode,
                                        HloInstruction* operand) override;
  Status HandleElementwiseBinary(HloInstruction* hlo, HloOpcode opcode,
                                         HloInstruction* lhs,
                                         HloInstruction* rhs) override;
  Status HandleClamp(HloInstruction* clamp, HloInstruction* min,
                             HloInstruction* arg, HloInstruction* max) override;
  Status HandleSelect(HloInstruction* select, HloInstruction* pred,
                              HloInstruction* on_true,
                              HloInstruction* on_false) override;
  Status HandleConcatenate(
          HloInstruction* concatenate,
          tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;

  Status HandleDot(HloInstruction* dot, HloInstruction* lhs,
                           HloInstruction* rhs) override;

  Status HandleConvolution(HloInstruction* convolution,
                                   HloInstruction* lhs, HloInstruction* rhs,
                                   const Window& window) override;
  Status HandleCrossReplicaSum(HloInstruction* crs) override;

  Status HandleInfeed(HloInstruction* infeed) override;
  Status HandleOutfeed(HloInstruction* outfeed) override;
  Status HandleRng(HloInstruction* random,
                           RandomDistribution distribution) override;
  Status HandleReverse(HloInstruction* reverse,
                               HloInstruction* operand) override;
  Status HandleSort(HloInstruction* sort, HloInstruction* operand) override;
  Status HandleConstant(HloInstruction* constant,
                                const Literal& literal) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                                       HloInstruction* operand) override;
  Status HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                              HloInstruction* init_value,
                              tensorflow::gtl::ArraySlice<int64> dimensions,
                              HloComputation* function) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleBroadcast(HloInstruction* broadcast) override;
  Status HandleReshape(HloInstruction* reshape) override;
  Status HandleTranspose(HloInstruction* transpose) override;
  Status HandleParameter(HloInstruction* parameter) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleCustomCall(
          HloInstruction* custom_call,
          tensorflow::gtl::ArraySlice<HloInstruction*> operands,
          tensorflow::StringPiece custom_call_target) override;
  Status HandleSlice(HloInstruction* slice,
                             HloInstruction* operand) override;
  Status HandleDynamicSlice(
          HloInstruction* slice,
          tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleDynamicUpdateSlice(HloInstruction* dynamic_update_slice,
                                  HloInstruction* operand,
                                  HloInstruction* update,
                                  HloInstruction* start_indices) override;
  Status HandleTuple(
          HloInstruction* tuple,
          tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleMap(
          HloInstruction* map,
          tensorflow::gtl::ArraySlice<HloInstruction*> operands,
          HloComputation* function,
          tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) override;
  Status HandleReduceWindow(HloInstruction* reduce_window,
                                    HloInstruction* operand,
                                    const Window& window,
                                    HloComputation* function) override;
  Status HandleSelectAndScatter(HloInstruction* instruction) override;
  Status HandleWhile(HloInstruction* xla_while) override;

  Status HandlePad(HloInstruction* pad) override;

  Status HandleSend(HloInstruction* send) override;

  Status HandleRecv(HloInstruction* recv) override;

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  Status FinishVisit(HloInstruction* root) override;
  
};

}  // namespace exampleplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_EXAMPLE_VISITOR_H_
