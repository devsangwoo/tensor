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
#include "tensorflow/core/graph/subgraph.h"

#include <algorithm>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
<<<<<<< HEAD
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace subgraph {
=======
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// ----------------------------------------------------------------------------
// Subgraph construction-related routines
// ----------------------------------------------------------------------------
// TODO(vrv): Profile the unordered_set and unordered_map use in this file to
// see if we should use an alternative implementation.

namespace {

<<<<<<< HEAD
typedef std::unordered_map<StringPiece, Node*, StringPieceHasher> NameIndex;
=======
typedef std::unordered_map<StringPiece, Node*, StringPiece::Hasher> NameIndex;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

// Rewrite graph by replacing the output tensors specified in
// "fed_outputs" with special feed nodes for each specified output
// tensor, and removing any nodes that are now disconnected from the
// part of the graph that reaches the sink node.  The set of special
// feed nodes added to the graph are returned in "*feed_nodes".
//
// Return true on success.  On error, return false and sets *error to
// an appropriate error message (and *g is left in an indeterminate
// state).
<<<<<<< HEAD
Status FeedInputs(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
    NameIndex* name_index, DataTypeVector* out_feed_types) {
  out_feed_types->clear();
  out_feed_types->reserve(feed_rewrites.size());
  for (size_t i = 0; i < feed_rewrites.size(); ++i) {
    const string& t = feed_rewrites[i]->endpoint_name();
=======
static Status FeedInputs(Graph* g, const DeviceAttributes& device_info,
                              const gtl::ArraySlice<string>& fed_outputs,
                              NameIndex* name_index) {
  for (const string& t : fed_outputs) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    TensorId id(ParseTensorName(t));

    auto iter = name_index->find(id.first);
    if (iter == name_index->end()) {
      return errors::NotFound("FeedInputs: unable to find feed output ", t);
    }
<<<<<<< HEAD
    Node* n = iter->second;
=======
    const Node* n = iter->second;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    DCHECK_EQ(n->name(), id.first);
    if (id.second >= n->num_outputs()) {
      return errors::InvalidArgument(
          "FeedInputs: ", t, " should have output index < ", n->num_outputs());
    }

<<<<<<< HEAD
    Node* feed_node;
    TF_RETURN_IF_ERROR(
        feed_rewrites[i]->AddNode(g, {n, id.second}, &feed_node));

    // Update name_index
    (*name_index)[feed_node->name()] = feed_node;
    // Duplicate control edges aren't allowed, but feed_node was *just* created
    // so there's no need to check for a duplicate.
    g->AddControlEdge(g->source_node(), feed_node, true);
=======
    Node* recv_node;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat("_recv_", id.first, "_", id.second),
                    "_Recv")
            .Attr("tensor_type", BaseType(n->output_type(id.second)))
            .Attr("tensor_name", t)
            .Attr("send_device", device_info.name())
            .Attr("recv_device", device_info.name())
            .Attr("send_device_incarnation",
                  static_cast<int64>(device_info.incarnation()))
            .Attr("client_terminated", true)
            .Finalize(g, &recv_node));
    recv_node->set_assigned_device_name(device_info.name());

    // Update name_index
    (*name_index)[recv_node->name()] = recv_node;
    g->AddControlEdge(g->source_node(), recv_node);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    // Look through edges coming out of "n" for edges whose src_output() index
    // matches "output_index".  If found, replace the edges with a connection
    // from the special feed node.
    std::vector<const Edge*> to_remove;
    for (const Edge* e : n->out_edges()) {
      if (e->src_output() == id.second) {
        to_remove.emplace_back(e);
      } else if (e->src_output() == Graph::kControlSlot &&
<<<<<<< HEAD
                 (n->type_string() == "Placeholder" ||
                  n->type_string() == "PlaceholderV2")) {
        // When feeding a Placeholder node, any outgoing control edges
        // will be replaced with a control edge from the replacement
        // feed_node.
=======
                 n->def().op() == "Placeholder") {
        // When feeding a Placeholder node, any outgoing control edges
        // will be replaced with a control edge from the replacement
        // recv_node.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
        // TODO(josh11b,mrry): Come up with a more elegant way of addressing
        // the general version of this problem.
        to_remove.emplace_back(e);
      }
    }

    for (const Edge* e : to_remove) {
      if (e->src_output() == id.second) {
<<<<<<< HEAD
        g->AddEdge(feed_node, 0, e->dst(), e->dst_input());
      } else {
        CHECK_EQ(Graph::kControlSlot, e->src_output());
        // Duplicate control edges aren't allowed, but feed_node was *just*
        // created so there's no need to check for a duplicate.
        g->AddControlEdge(feed_node, e->dst(), true);
      }
      g->RemoveEdge(e);
    }
    out_feed_types->push_back(BaseType(n->output_type(id.second)));
=======
        g->AddEdge(recv_node, 0, e->dst(), e->dst_input());
      } else {
        CHECK_EQ(Graph::kControlSlot, e->src_output());
        g->AddControlEdge(recv_node, e->dst());
      }
      g->RemoveEdge(e);
    }
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }
  return Status::OK();
}

<<<<<<< HEAD
Status FetchOutputs(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
    NameIndex* name_index, std::vector<Node*>* out_fetch_nodes,
    DataTypeVector* out_fetch_types) {
  out_fetch_nodes->clear();
  out_fetch_nodes->reserve(fetch_rewrites.size());
  for (size_t i = 0; i < fetch_rewrites.size(); ++i) {
    const string& t = fetch_rewrites[i]->endpoint_name();

=======
// Augment "*g" by adding special "fetch" nodes that connect to the
// tensor outputs specified in "fetch_outputs" to retrieve the output
// of the tensors.  The new nodes added are set up to execute on
// "client_device_name", and are returned in "*fetch_nodes".
//
// Return true on success.  On error, return false and sets *error to
// an appropriate error message (and *g is left in an indeterminate
// state).
static Status FetchOutputs(Graph* g, const DeviceAttributes& device_info,
                           const gtl::ArraySlice<string>& fetch_outputs,
                           NameIndex* name_index,
                           std::vector<Node*>* fetch_nodes) {
  fetch_nodes->clear();
  for (const string& t : fetch_outputs) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    // Parse t into node_name and output_index.
    TensorId id(ParseTensorName(t));

    // Find node in graph with that name.
    auto iter = name_index->find(id.first);
    if (iter == name_index->end()) {
      return errors::NotFound("FetchOutputs node ", t, ": not found");
    }
    Node* n = iter->second;
    DCHECK_EQ(n->name(), id.first);
    VLOG(2) << "Found fetch node for " << t;

    // Validate output_index
<<<<<<< HEAD
    if (n->num_outputs() == 0) {
      return errors::InvalidArgument(
          "Tried to fetch data for '", t,
          "', which produces no output.  To run to a node but not fetch any "
          "data, pass '",
          t,
          "' as an argument to the 'target_node_names' argument of the "
          "Session::Run API.");
    } else if (id.second >= n->num_outputs()) {
=======
    if (id.second >= n->num_outputs()) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      return errors::InvalidArgument("FetchOutputs ", t,
                                     ": output index too large, must be < ",
                                     n->num_outputs());
    }

    // Create the fetch Node and connect it up
<<<<<<< HEAD
    Node* fetch_node;
    TF_RETURN_IF_ERROR(
        fetch_rewrites[i]->AddNode(g, {n, id.second}, &fetch_node));

    // Update the index.
    (*name_index)[fetch_node->name()] = fetch_node;

    // Duplicate control edges aren't allowed, but fetch_node was *just* created
    // so there's no need to check for a duplicate.
    g->AddControlEdge(fetch_node, g->sink_node(), true);
    out_fetch_nodes->push_back(fetch_node);
    out_fetch_types->push_back(BaseType(n->output_type(id.second)));
=======
    Node* send_node;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat("_send_", id.first, "_", id.second),
                    "_Send")
            .Input(n, id.second)
            .Attr("tensor_name", t)
            .Attr("send_device", device_info.name())
            .Attr("recv_device", device_info.name())
            .Attr("send_device_incarnation",
                  static_cast<int64>(device_info.incarnation()))
            .Attr("client_terminated", true)
            .Finalize(g, &send_node));
    send_node->set_assigned_device_name(device_info.name());
    VLOG(1) << "Created fetch node: " << SummarizeNodeDef(send_node->def());

    // Update the index.
    (*name_index)[send_node->name()] = send_node;

    g->AddControlEdge(send_node, g->sink_node());
    fetch_nodes->push_back(send_node);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  return Status::OK();
}

<<<<<<< HEAD
bool AddNodeToTargets(const string& node_or_tensor_name,
                      const NameIndex& name_index,
                      std::unordered_set<const Node*>* targets) {
=======
static bool AddNodeToTargets(const string& node_or_tensor_name,
                             const NameIndex& name_index,
                             std::unordered_set<const Node*>* targets) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  TensorId id = ParseTensorName(node_or_tensor_name);
  auto iter = name_index.find(id.first);
  if (iter == name_index.end()) {
    return false;
  }
  const Node* n = iter->second;
<<<<<<< HEAD
  CHECK_EQ(n->name(), id.first);
=======
  if (n->name() != node_or_tensor_name) {
    return false;
  }

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  targets->insert(n);
  return true;
}

<<<<<<< HEAD
Status PruneForTargets(Graph* g, const NameIndex& name_index,
                       const std::vector<Node*>& fetch_nodes,
                       const gtl::ArraySlice<string>& target_nodes) {
=======
static Status PruneForTargets(Graph* g, const NameIndex& name_index,
                              const std::vector<Node*>& fetch_nodes,
                              const gtl::ArraySlice<string>& target_nodes) {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  string not_found;
  std::unordered_set<const Node*> targets;
  for (Node* n : fetch_nodes) {
    if (!AddNodeToTargets(n->name(), name_index, &targets)) {
      strings::StrAppend(&not_found, n->name(), " ");
    }
  }
  for (const string& s : target_nodes) {
    if (!AddNodeToTargets(s, name_index, &targets)) {
      strings::StrAppend(&not_found, s, " ");
    }
  }
  if (!not_found.empty()) {
    return errors::NotFound("PruneForTargets: Some target nodes not found: ",
                            not_found);
  }
<<<<<<< HEAD
  PruneForReverseReachability(g, std::move(targets));

  // Reconnect nodes with no outgoing edges to the sink node
  FixupSourceAndSinkEdges(g);
=======
  PruneForReverseReachability(g, targets);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  return Status::OK();
}

}  // namespace

<<<<<<< HEAD
Status ArgFeedRewrite::AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                               Node** out_node) {
  // NOTE(mrry): We must include the index as part of the node
  // name, because _Arg is a "stateful" kernel and therefore
  // its name must uniquely identify a kernel instance across all
  // graphs in the same session.
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_arg_", feed_tensor.node->name(), "_",
                                  feed_tensor.index, "_", arg_index_),
                  "_Arg")
          .Attr("T", BaseType(feed_tensor.node->output_type(feed_tensor.index)))
          .Attr("index", arg_index_)
          .Finalize(g, out_node, /*consume=*/true));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status RecvFeedRewrite::AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                                Node** out_node) {
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_recv_", feed_tensor.node->name(), "_",
                                  feed_tensor.index),
                  "_Recv")
          .Attr("tensor_type",
                BaseType(feed_tensor.node->output_type(feed_tensor.index)))
          .Attr("tensor_name", endpoint_name())
          .Attr("send_device", device_info().name())
          .Attr("recv_device", device_info().name())
          .Attr("send_device_incarnation",
                static_cast<int64>(device_info().incarnation()))
          .Attr("client_terminated", true)
          .Finalize(g, out_node, /*consume=*/true));

  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status RetvalFetchRewrite::AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                                   Node** out_node) {
  // NOTE(mrry): We must include the index as part of the node
  // name, because _Retval is a "stateful" kernel and therefore
  // its name must uniquely identify a kernel instance across all
  // graphs in the same session.
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_retval_", fetch_tensor.node->name(), "_",
                                  fetch_tensor.index, "_", retval_index_),
                  "_Retval")
          .Input(fetch_tensor.node, fetch_tensor.index)
          .Attr("T",
                BaseType(fetch_tensor.node->output_type(fetch_tensor.index)))
          .Attr("index", retval_index_)
          .Finalize(g, out_node, /*consume=*/true));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status SendFetchRewrite::AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                                 Node** out_node) {
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_send_", fetch_tensor.node->name(), "_",
                                  fetch_tensor.index),
                  "_Send")
          .Input(fetch_tensor.node, fetch_tensor.index)
          .Attr("tensor_name", endpoint_name())
          .Attr("send_device", device_info().name())
          .Attr("recv_device", device_info().name())
          .Attr("send_device_incarnation",
                static_cast<int64>(device_info().incarnation()))
          .Attr("client_terminated", true)
          .Finalize(g, out_node, /*consume=*/true));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}
=======
namespace subgraph {
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

Status RewriteGraphForExecution(
    Graph* g, const gtl::ArraySlice<string>& fed_outputs,
    const gtl::ArraySlice<string>& fetch_outputs,
    const gtl::ArraySlice<string>& target_node_names,
<<<<<<< HEAD
    const DeviceAttributes& device_info, bool use_function_convention,
    RewriteGraphMetadata* out_metadata) {
  std::vector<std::unique_ptr<PruneRewrite>> feed_rewrites;
  feed_rewrites.reserve(fed_outputs.size());
  if (use_function_convention) {
    for (size_t i = 0; i < fed_outputs.size(); ++i) {
      feed_rewrites.emplace_back(new ArgFeedRewrite(
          &fed_outputs[i], &device_info, static_cast<int32>(i)));
    }
  } else {
    for (const string& fed_output : fed_outputs) {
      feed_rewrites.emplace_back(
          new RecvFeedRewrite(&fed_output, &device_info));
    }
  }

  std::vector<std::unique_ptr<PruneRewrite>> fetch_rewrites;
  fetch_rewrites.reserve(fetch_outputs.size());
  if (use_function_convention) {
    for (size_t i = 0; i < fetch_outputs.size(); ++i) {
      fetch_rewrites.emplace_back(new RetvalFetchRewrite(
          &fetch_outputs[i], &device_info, static_cast<int32>(i)));
    }
  } else {
    for (const string& fetch_output : fetch_outputs) {
      fetch_rewrites.emplace_back(
          new SendFetchRewrite(&fetch_output, &device_info));
    }
  }

  return RewriteGraphForExecution(g, feed_rewrites, fetch_rewrites,
                                  target_node_names, out_metadata);
}

namespace {
template <typename StringContainer>
std::vector<string> ConvertToVector(StringContainer field) {
  return std::vector<string>(field.begin(), field.end());
}
}  // namespace

Status RewriteGraphForExecution(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
    const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
    const gtl::ArraySlice<string>& target_node_names,
    RewriteGraphMetadata* out_metadata) {
  if (fetch_rewrites.empty() && target_node_names.empty()) {
    return errors::InvalidArgument(
        "Must specify at least one target to fetch or execute.");
  }

  std::unordered_set<string> endpoints;
  for (const auto& feed_rewrite : feed_rewrites) {
    auto result = endpoints.insert(feed_rewrite->endpoint_name());
    if (!result.second) {
      return errors::InvalidArgument("Endpoint \"",
                                     feed_rewrite->endpoint_name(),
                                     "\" fed more than once.");
    }
  }

  for (const auto& fetch_rewrite : fetch_rewrites) {
    if (endpoints.count(fetch_rewrite->endpoint_name()) > 0) {
      return errors::InvalidArgument(fetch_rewrite->endpoint_name(),
                                     " is both fed and fetched.");
=======
    const DeviceAttributes& device_info) {
  std::unordered_set<string> endpoints(fed_outputs.begin(), fed_outputs.end());
  for (const auto& fetch : fetch_outputs) {
    if (endpoints.count(fetch) > 0) {
      return errors::InvalidArgument(fetch, " is both fed and fetched.");
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    }
  }

  // A separate index mapping name to Node*, for use by FeedInputs,
  // FetchOutputs, and PruneForTargets
  NameIndex name_index;
<<<<<<< HEAD
  name_index.reserve(g->num_nodes());
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  for (Node* n : g->nodes()) {
    name_index[n->name()] = n;
  }

  // Add the feeds.  This may replace nodes in the graph, including the nodes
<<<<<<< HEAD
  // currently listed in "fetch_rewrites".  We pass "name_index" so the index is
  // kept up to date.
  if (!feed_rewrites.empty()) {
    TF_RETURN_IF_ERROR(
        FeedInputs(g, feed_rewrites, &name_index, &out_metadata->feed_types));
=======
  // currently listed in "fetch_nodes".  We pass "name_index" so the index is
  // kept up to date.
  if (!fed_outputs.empty()) {
    TF_RETURN_IF_ERROR(FeedInputs(g, device_info, fed_outputs, &name_index));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  // Add the fetch nodes, also updating "name_index".
  std::vector<Node*> fetch_nodes;
<<<<<<< HEAD
  if (!fetch_rewrites.empty()) {
    TF_RETURN_IF_ERROR(FetchOutputs(g, fetch_rewrites, &name_index,
                                    &fetch_nodes, &out_metadata->fetch_types));
  }

  // Prune the graph to only compute what is needed for the fetch nodes and the
  // target nodes.
=======
  if (!fetch_outputs.empty()) {
    TF_RETURN_IF_ERROR(
        FetchOutputs(g, device_info, fetch_outputs, &name_index, &fetch_nodes));
  }

  // Prune the graph to only compute what is needed for the fetch nodes and the
  // targets nodes.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  if (!fetch_nodes.empty() || !target_node_names.empty()) {
    TF_RETURN_IF_ERROR(
        PruneForTargets(g, name_index, fetch_nodes, target_node_names));
  }

  return Status::OK();
}

}  // namespace subgraph

}  // namespace tensorflow
