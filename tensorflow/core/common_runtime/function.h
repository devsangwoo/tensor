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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_H_

#include <functional>
#include <memory>

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

static constexpr const char* const kNoInlineAttr = "_noinline";

// Get default customizable kernel creator if set
const CustomKernelCreator* GetDefaultCustomKernelCreator();

// Registers a default customizable kernel creator for a function call.
//
// If c->CanCreateKernel returns false, we still fall back to an executor-based
// interpreter op kernel to execute a function. Else c->CreateKernel() can be
// used to create a kernel that will compile the function with XLA and run the
// resulting program.
//
// TODO(zhifengc/phawkins): b/32379046
void RegisterDefaultCustomKernelCreator(CustomKernelCreator* c);

// Creates a FunctionLibraryRuntime, which instantiates functions
// defined in "lib_def" and executes functions on the "device".
// "device_mgr" must contain the "device". If not nullptr,
// "custom_kernel_creator" is consulted by the returned runtime to
// create kernels.
=======
#ifndef TENSORFLOW_COMMON_RUNTIME_FUNCTION_H_
#define TENSORFLOW_COMMON_RUNTIME_FUNCTION_H_

#include <functional>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Creates a FunctionLibraryRuntime, which instantiates functions
// defined in "lib_def" and executes functions on the "device".
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
//
// The returned object does not take ownerships of "device" or
// "lib_def".  The caller must ensure "device" and "lib_def" outlives
// the returned object.
<<<<<<< HEAD
//
// The "parent" is a pointer to the ProcessFunctionLibraryRuntime object that
// typically owns the created FunctionLibraryRuntime object. The parent pointer
// is not owned by the FunctionLibraryRuntime object.
std::unique_ptr<FunctionLibraryRuntime> NewFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, const ConfigProto* config,
    Device* device, int graph_def_version,
    const FunctionLibraryDefinition* lib_def, thread::ThreadPool* thread_pool,
    const OptimizerOptions& optimizer_options,
    const CustomKernelCreator* custom_kernel_creator,
    const SessionMetadata* session_metadata,
    ProcessFunctionLibraryRuntime* parent);
=======
typedef std::function<void()> Closure;
typedef std::function<void(Closure)> Runner;
FunctionLibraryRuntime* NewFunctionLibraryRuntime(
    Device* device, Runner runner, const FunctionLibraryDefinition* lib_def);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// FunctionLibraryRuntime::GetFunctionBody returns a description of an
// instantiated function that is represented as a Graph with arg/ret
// nodes annotated.
struct FunctionBody {
  FunctionDef fdef;
  Graph* graph = nullptr;  // owned.
  DataTypeVector arg_types;
  DataTypeVector ret_types;
<<<<<<< HEAD
  // arg_nodes[i] contains the i'th function input. In other words,
  // GetNodeAttr(arg_nodes[i]->attrs(), "index") == i.
  gtl::InlinedVector<Node*, 4> arg_nodes;
  // ret_nodes[i] contains the i'th function output. In other words,
  // GetNodeAttr(ret_nodes[i]->attrs(), "index") == i.
  gtl::InlinedVector<Node*, 4> ret_nodes;
  gtl::InlinedVector<Node*, 4> control_ret_nodes;
=======
  gtl::InlinedVector<Node*, 4> arg_nodes;
  gtl::InlinedVector<Node*, 4> ret_nodes;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  FunctionBody() {}
  FunctionBody(const FunctionDef& f, DataTypeSlice arg_types,
               DataTypeSlice ret_types, Graph* g);
  ~FunctionBody();
};

// Debugging facility.  Returns a debug string for a graph
// representing an instantiated function.
<<<<<<< HEAD
string DebugString(const Graph* g);
=======
string DebugString(const Graph* instantiated_func_graph);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// A few hand-crafted optimization on the instantiated function body
// (a Graph*).

// Removes nodes that are
//   1. not stateful; and
//   2. not _Arg; and
//   3. not reachable from _Retval.
<<<<<<< HEAD
//
// This function is triggered by function inlining, unlike 'PruneFunctionBody'
// it doesn't preserve nodes that are reachable from control returns. Function
// inlining is responsible for connecting control return nodes with the nodes
// that have input control edges from the inlined function call node.
//
// Assuming that automatic control dependency tracking is correct, absence of
// outgoing control edge from the function call node means that no one needs to
// observe side-effect that might have been generated by the function (see
// documentation in common_runtime/function.cc for details).
//
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// Returns true iff any node is removed from "g".
bool RemoveDeadNodes(Graph* g);

// Find a pattern:
//   src -(in)-> node -(out)-> dst, where
// 1) node is an identity node;
// 2) in is the only incoming data edge;
// 3) out is the only outgoing data edge;
//
// Rewrites the above pattern with src->dst and relevant data
// dependencies updated. Repeat the process until no such pattern
// left.
bool RemoveIdentityNodes(Graph* g);

// Rewrites _ListToArray and _ArrayToList to a set of Identity nodes.
bool RemoveListArrayConverter(Graph* g);

<<<<<<< HEAD
// Dump the contents of the "graph" to log files if the logging level is
// sufficiently high.
void DumpGraph(StringPiece label, const Graph* g);

// Applies graph rewrite optimization such as inlining, dead code
=======
// For each node in "graph", if "lib" indicates that the node is a
// function call, inline the function body.  Returns true if at least
// one node is inlined.
//
// This routine goes through "graph" nodes once and applies the
// inlining.  The caller may decide to apply the inlining on "graph"
// multiple times by calling ExpandInlineFunctions a few times.
bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph);

// Applies graph rewrite optimzation such as inlining, dead code
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// removal, etc.
//
// **g is a graph constructed based on the runtime library 'lib'.
// OptimizeGraph mutates **g extensively and replaces '*g' with a
// complete copy. Therefore, the caller should not keep any references
// to nodes *g.
<<<<<<< HEAD
void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g,
                   const GraphOptimizer::Options& graph_optimizer_options);
void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g);

// Convert the Graph of a function to a GraphDef.
//
// Handles renaming of nodes to avoid duplicate names which may
// be present after various rewriting operations.
void ToGraphDef(const Graph* g, GraphDef* gdef, bool pretty = false);
=======
void OptimizeGraph(FunctionLibraryRuntime* lib, Graph** g);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// Given a numerical function "f", returns another numerical function
// "g", such that if "f" takes N inputs and produces M outputs, "g"
// takes N + M inputs and produces N outputs. I.e., if
//   (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
// g is a function which is
//   (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
//                                     dL/dy1, dL/dy2, ..., dL/dy_M),
// where L is a scalar-value function of (...x_i...).
//
// TODO(zhifengc): Asks math expert to say the comment again.
<<<<<<< HEAD
std::unique_ptr<FunctionBody> SymbolicGradient(const FunctionBody& f);

// Optionally override device assignment for nodes added to the graph for
// inlined functions:
// (1) Identity nodes added in place of function input arguments.
// (2) Identity nodes added in place of function return values.
// (3) Special NoOp nodes that enforce side-effects execution order.
// (4) All nodes inside function body specified in FunctionDef.
class InlinedFunctionBodyPlacer {
 public:
  virtual ~InlinedFunctionBodyPlacer() = default;

  virtual absl::optional<string> InputNodeDevice(int input_index) const = 0;
  virtual absl::optional<string> OutputNodeDevice(int output_index) const = 0;
  // Returns true if the added input/output identity nodes should be colocated
  // with the corresponding input/output from the function body.
  virtual bool ColocateInputOutputIdentities() const = 0;
  virtual absl::optional<string> ControlNodeDevice() const = 0;
  virtual absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const = 0;

  // Place input nodes on the same device as the corresponding caller input
  // node. Do not specify any placement for all other nodes.
  static std::unique_ptr<InlinedFunctionBodyPlacer> DefaultPlacer(
      const Graph& graph, const Node& caller);

  // Place all nodes on the same device as caller node.
  static std::unique_ptr<InlinedFunctionBodyPlacer> SingleDevicePlacer(
      const Graph& graph, const Node& caller);

  // Place input nodes on the same device as the corresponding caller input
  // node. Do not place output node. Place control nodes on the same device as
  // caller node. For all function body nodes overrides job, replica and task
  // parts of the device assignment to match function caller node.
  static std::unique_ptr<InlinedFunctionBodyPlacer> MultiDevicePlacer(
      const Graph& graph, const Node& caller);

  using Factory = std::function<std::unique_ptr<InlinedFunctionBodyPlacer>(
      const Graph&, const Node&)>;

  struct Config {
    string name;
    Factory get;
  };

  static Config Default() { return {"default", DefaultPlacer}; }
  static Config SingleDevice() { return {"single_device", SingleDevicePlacer}; }
  static Config MultiDevice() { return {"multi_device", MultiDevicePlacer}; }
};

struct InlineFunctionBodyOptions {
  // All nodes that have incoming control edge *from* the function call node,
  // will be forwarded to the "output control node". There are two options for
  // choosing which nodes will have a control edge *to* the "output control
  // node":
  //   a) control returns            (`control_ret` field in FunctionDef)
  //   b) data returns               (`ret` field in FunctionDef)
  enum class OutputControlSource { kDataOutputs, kControlOutputs };

  // Keep a node in a graph with the same name as the function call node:
  //
  // a) DoNotKeep: Function call node is fully inlined, and there is no node in
  //    a graph with the same name.
  //
  // b) Fetchable: Add an IdentityN node to the graph in place of the inlined
  //    function call node. It will have a control edge from inlined
  //    'output_control_node' and data edges from function output nodes.
  //    The IdentityN node will be placed on the same device as the caller node.
  //
  //    This is mostly for compatibility with Tensorflow v1 and sessions.
  //    When we prepare a graph for execution in
  //    GraphExecutionState::MakeForBaseGraph we don't know what nodes will be
  //    fetched, so we can't safely remove any of them. When graph executed as a
  //    function it has 'Retval' nodes for all fetched tensors, and we can
  //    safely inline function calls.
  //
  // c) Targetable: Add a NoOp node to the graph in place of the inlined
  //    function call node. It will have a control edge from inline
  //    'output_control_node' and no data edges. NoOp node will be placed on the
  //    same device as the caller node. This will keep the inlined function call
  //    node a valid 'session.run' target, and also will keep it a valid control
  //    output node.
  enum class KeepCallerNode { kDoNotKeep, kFetchable, kTargetable };

  // If 'true' function inlining is completely disabled. This allows to control
  // function inlining for different types of function calls (see
  // 'ExpandInlineFunctionsOptions' below).
  bool disable_inlining = false;
  // Ignore '_noinline' function attribute.
  bool ignore_noinline = false;
  // If 'true' function inlining will inline functions in implementation
  // selection group. Normally those functions should not be inlined; they will
  // be handled by Grappler.
  bool inline_impl_selection_group_functions = false;
  // Controls if we want to keep a node with the name as the function call node
  // in a graph after function inlining.
  KeepCallerNode keep_caller_node = KeepCallerNode::kDoNotKeep;
  // For compatibility with Tensorflow v1 by default we will use data outputs.
  // Control returns were added to Tensorflow v2 with automatic control
  // dependencies tracking in Eager mode.
  OutputControlSource output_control_src = OutputControlSource::kDataOutputs;
  // Inlined function body placer decides what requested device assignments
  // should be added to the nodes added to the graph. See documentation above
  // for available strategies.
  InlinedFunctionBodyPlacer::Config inlined_function_body_placer =
      InlinedFunctionBodyPlacer::Default();
  // If true, frame names in the function body will be
  // made unique in the resulting graph (e.g. by prepending a unique prefix).
  // NOTE(mrry): Only set this option to false when there is a single function
  // call in the graph (e.g. when making a remote function call via
  // ClusterFunctionLibraryRuntime). This option is provided because the graph
  // partitioner generates frame names that must remain unmodified across all
  // partitions of a multi-device function.
  bool uniquify_frame_names = true;

  // A human-readable debug string for this options.
  string DebugString() const;
};

// Returns 'Status::OK()' iff the function '*fbody' can be inlined at 'node'
// based on the type signature of 'node' and 'fbody':
//
// (1) Caller node has the same number of inputs and outputs as the function.
// (2) Caller node inputs and outputs have the same data types as function
//     inputs and returns.
// (3) Validation rules defined in InlineFunctionBodyOptions.
//
// If function can't be safely inlined, returns error message with details why
// inlining is not possible or safe.
Status ValidateInlining(const Node* node, const FunctionBody* fbody,
                        const InlineFunctionBodyOptions& options);

// Given a "caller" in graph "g", which is a function call of a function
// to "fbody". Replaces the "caller" with fbody->graph and connects
// edges properly. "override_device" specifies whether inlining should replace
// explicitly specified devices inside fbody with the callee's device.
//
// Returns 'Status::OK()' if function was successfully inlined into the graph.
// If function inlining is not possible returns an error with a reason, and
// leaves the graph in unmodified state.
Status InlineFunctionBody(const FunctionLibraryDefinition& flib_def, Graph* g,
                          Node* caller, const FunctionBody* fbody,
                          const InlineFunctionBodyOptions& options);

// There are three types of function calls that could be invoked during
// *Tensorflow graph execution*:
//
// 1) Native function call (node.type_string() is the function name). These
//    functions are always executed on a single-device, which is the device of
//    the function call node.
//
// 2) Multi-device function calls (PartitionedCall or StatefulPartitionedCall
//    ops) can execute on multiple devices and accept DT_RESOURCE inputs that
//    belong to different devices. This type of functions was added in
//    Tensorflow 2.0 Eager mode, and it has control outputs to represent
//    side-effects that must always execute (see `control_ret` in FunctionDef).
//
// 3) SymbolicGradient has been deprecated for a while, but we still keep it and
//    use `native` options for inlining for compatibility.
//
// We need to have distinct inlining rules for compatibility with Tensorflow v1.
//
// There are few other places in Tensorflow that could execute functions:
//
// 1) common_runtime/eager/kernel_and_device.{h,cc} - executes "top level"
//    functions directly via function library runtime, without going through
//    the graph.
// 2) tf.data pipelines - also execute functions directly via function library
//    runtime with custom executors.
struct ExpandInlineFunctionsOptions {
  ExpandInlineFunctionsOptions() : native_options(), multi_device_options() {
    using OutputControlSrc = InlineFunctionBodyOptions::OutputControlSource;
    multi_device_options.output_control_src = OutputControlSrc::kControlOutputs;
  }

  InlineFunctionBodyOptions native_options;
  InlineFunctionBodyOptions multi_device_options;
};

// WARNING(ezhulenev): PLEASE DO NOT USE THIS FUNCTION. This is a temporary
// workaround that will be enabled only during the function inlining unification
// (b/126811947). Contact ezhulenev@ if you think you need it.
// TODO(ezhulenev): Delete this function.
bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph,
                           const ExpandInlineFunctionsOptions& options);

// For each node in "graph", if "lib" indicates that the node is a
// function call, inline the function body. Returns true if at least
// one node is inlined.
//
// This routine goes through "graph" nodes once and applies the
// inlining. The caller may decide to apply the inlining on "graph"
// multiple times by calling ExpandInlineFunctions a few times.
//
// Function calls that can't be safely inlined into the graph (ValidateInlining
// returns error), are ignored.
//
// TODO(ezhulenev): We do not FunctionLibraryRuntime for this. We need just the
// FunctionLibraryDefinition and FunctionDefToBodyHelper to implement this (see
// lower_function_call.cc).
inline bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph) {
  return ExpandInlineFunctions(lib, graph, ExpandInlineFunctionsOptions());
}

// Extracts function name and attributes from `call_def`
// `call_def` can be a native function call (where the op type is the function
// name) or a call through PartitionedCall/StatefulPartitionedCall.
Status NameAndAttrsFromFunctionCall(const NodeDef& call_def,
                                    NameAttrList* function);

// Extracts function name and attributes from `call_def` and invokes
// flr->Instantiate(name, attrs, handle).
// `call_def` can be a native function call (where the op type is the function
// name) or a call through PartitionedCall/StatefulPartitionedCall.
Status InstantiateFunctionCall(const NodeDef& call_def,
                               FunctionLibraryRuntime* flr,
                               FunctionLibraryRuntime::Handle* handle);

// Returns true iff `n` represents a function call. `n` can be a native
// function call (n.type_string() is the function name),
// a PartitionedCall/StatefulPartitionedCall, or a SymbolicGradient (which
// has been deprecated for a while).
bool IsFunctionCall(const FunctionLibraryDefinition& lib_def, const Node& n);

// Instantiates FunctionDef into a graph. Set *fbody to point to the
// FunctionBody that holds the instantiated FunctionDef.
Status FunctionDefToBodyHelper(const FunctionDef& fdef, const AttrSlice& attrs,
                               const FunctionLibraryDefinition* lib_def,
                               std::unique_ptr<FunctionBody>* fbody);

// Instantiates FunctionDef into a graph. Set *fbody to point to the
// FunctionBody that holds the instantiated FunctionDef. Use custom function
// signature lookup, in case instantiated function is not in the 'lib_def'.
Status FunctionDefToBodyHelper(
    const FunctionDef& fdef, const AttrSlice& attrs,
    const FunctionLibraryDefinition* lib_def,
    const std::function<Status(const string&, const OpDef**)>& get_func_sig,
    std::unique_ptr<FunctionBody>* fbody);
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_H_
=======
FunctionBody* SymbolicGradient(const FunctionBody& f);

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_FUNCTION_H_
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
