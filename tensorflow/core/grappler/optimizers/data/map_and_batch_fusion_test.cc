/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/map_and_batch_fusion.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

TEST(MapAndBatchFusionTest, FuseMapAndBatchNodesIntoOne) {
  GrapplerItem item;
  GraphDef *graph = &item.graph;
  NodeDef *start_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(0, graph, &start_node));
  NodeDef *stop_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(10, graph, &stop_node));
  NodeDef *step_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(1, graph, &step_node));

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node;
  TF_ASSERT_OK(graph_utils::AddNode("", "RangeDataset", range_inputs,
                                    range_attrs, graph, &range_node));
  NodeDef *captured_input_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<StringPiece>(
      "hello", graph, &captured_input_node));

  NodeDef *map_node;
  {
    std::vector<string> map_inputs(2);
    map_inputs[0] = range_node->name();
    map_inputs[1] = captured_input_node->name();
    std::vector<std::pair<string, AttrValue>> map_attrs(2);
    AttrValue f_attr;
    SetAttrValue("f", &f_attr);
    map_attrs[0] = std::make_pair("f", f_attr);
    AttrValue args_attr;
    SetAttrValue("Targuments", &args_attr);
    map_attrs[1] = std::make_pair("Targuments", args_attr);
    TF_ASSERT_OK(graph_utils::AddNode("", "MapDataset", map_inputs, map_attrs,
                                      graph, &map_node));
  }

  NodeDef *batch_size_node;
  TF_ASSERT_OK(
      graph_utils::AddScalarConstNode<int64>(5, graph, &batch_size_node));
  NodeDef *batch_node;
  {
    std::vector<string> batch_inputs(2);
    batch_inputs[0] = map_node->name();
    batch_inputs[1] = batch_size_node->name();
    std::vector<std::pair<string, AttrValue>> batch_attrs(2);
    AttrValue shapes_attr;
    SetAttrValue("output_shapes", &shapes_attr);
    batch_attrs[0] = std::make_pair("output_shapes", shapes_attr);
    AttrValue types_attr;
    SetAttrValue("output_types", &types_attr);
    batch_attrs[1] = std::make_pair("output_types", types_attr);
    TF_ASSERT_OK(graph_utils::AddNode("", "BatchDataset", batch_inputs,
                                      batch_attrs, graph, &batch_node));
  }

  MapAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsNodeWithName(map_node->name(), output));
  EXPECT_FALSE(graph_utils::ContainsNodeWithName(batch_node->name(), output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("MapAndBatchDatasetV2", output));
  NodeDef map_and_batch_node =
      output.node(graph_utils::FindNodeWithOp("MapAndBatchDatasetV2", output));
  EXPECT_EQ(map_and_batch_node.input_size(), 5);
  EXPECT_EQ(map_and_batch_node.input(0), map_node->input(0));
  EXPECT_EQ(map_and_batch_node.input(1), map_node->input(1));
  EXPECT_EQ(map_and_batch_node.input(2), batch_node->input(1));
  NodeDef num_parallel_calls_node = output.node(
      graph_utils::FindNodeWithName(map_and_batch_node.input(3), output));
  EXPECT_EQ(num_parallel_calls_node.attr().at("value").tensor().int64_val(0),
            1);
  NodeDef drop_remainder_node = output.node(
      graph_utils::FindNodeWithName(map_and_batch_node.input(4), output));
  EXPECT_EQ(drop_remainder_node.attr().at("value").tensor().bool_val(0), false);
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("f"),
                                 map_node->attr().at("f")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("Targuments"),
                                 map_node->attr().at("Targuments")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_shapes"),
                                 batch_node->attr().at("output_shapes")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_types"),
                                 batch_node->attr().at("output_types")));
}

TEST(MapAndBatchFusionTest, FuseParallelMapAndBatchNodesIntoOne) {
  GrapplerItem item;
  GraphDef *graph = &item.graph;
  NodeDef *start_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(0, graph, &start_node));
  NodeDef *stop_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(10, graph, &stop_node));
  NodeDef *step_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(1, graph, &step_node));

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node;
  TF_ASSERT_OK(graph_utils::AddNode("", "RangeDataset", range_inputs,
                                    range_attrs, graph, &range_node));
  NodeDef *captured_input_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<StringPiece>(
      "hello", graph, &captured_input_node));
  NodeDef *num_parallel_calls_node;
  TF_ASSERT_OK(
      graph_utils::AddScalarConstNode<int>(2, graph, &num_parallel_calls_node));

  NodeDef *map_node;
  {
    std::vector<string> map_inputs(3);
    map_inputs[0] = range_node->name();
    map_inputs[1] = captured_input_node->name();
    map_inputs[2] = num_parallel_calls_node->name();
    std::vector<std::pair<string, AttrValue>> map_attrs(2);
    AttrValue f_attr;
    SetAttrValue("f", &f_attr);
    map_attrs[0] = std::make_pair("f", f_attr);
    AttrValue args_attr;
    SetAttrValue("Targuments", &args_attr);
    map_attrs[1] = std::make_pair("Targuments", args_attr);
    TF_ASSERT_OK(graph_utils::AddNode("", "ParallelMapDataset", map_inputs,
                                      map_attrs, graph, &map_node));
  }

  NodeDef *batch_size_node;
  TF_ASSERT_OK(
      graph_utils::AddScalarConstNode<int64>(5, graph, &batch_size_node));
  NodeDef *batch_node;
  {
    std::vector<string> batch_inputs(2);
    batch_inputs[0] = map_node->name();
    batch_inputs[1] = batch_size_node->name();
    std::vector<std::pair<string, AttrValue>> batch_attrs(2);
    AttrValue shapes_attr;
    SetAttrValue("output_shapes", &shapes_attr);
    batch_attrs[0] = std::make_pair("output_shapes", shapes_attr);
    AttrValue types_attr;
    SetAttrValue("output_types", &types_attr);
    batch_attrs[1] = std::make_pair("output_types", types_attr);
    TF_ASSERT_OK(graph_utils::AddNode("", "BatchDataset", batch_inputs,
                                      batch_attrs, graph, &batch_node));
  }

  MapAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsNodeWithName(map_node->name(), output));
  EXPECT_FALSE(graph_utils::ContainsNodeWithName(batch_node->name(), output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("MapAndBatchDatasetV2", output));
  NodeDef map_and_batch_node =
      output.node(graph_utils::FindNodeWithOp("MapAndBatchDatasetV2", output));
  EXPECT_EQ(map_and_batch_node.input_size(), 5);
  EXPECT_EQ(map_and_batch_node.input(0), map_node->input(0));
  EXPECT_EQ(map_and_batch_node.input(1), map_node->input(1));
  EXPECT_EQ(map_and_batch_node.input(2), batch_node->input(1));
  NodeDef num_parallel_calls_node2 = output.node(
      graph_utils::FindNodeWithName(map_and_batch_node.input(3), output));
  EXPECT_EQ(num_parallel_calls_node2.attr().at("value").tensor().int64_val(0),
            2);
  NodeDef drop_remainder_node = output.node(
      graph_utils::FindNodeWithName(map_and_batch_node.input(4), output));
  EXPECT_EQ(drop_remainder_node.attr().at("value").tensor().bool_val(0), false);
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("f"),
                                 map_node->attr().at("f")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("Targuments"),
                                 map_node->attr().at("Targuments")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_shapes"),
                                 batch_node->attr().at("output_shapes")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_types"),
                                 batch_node->attr().at("output_types")));
}

TEST(MapAndBatchFusionTest, NoChange) {
  std::vector<std::pair<string, AttrValue>> empty_attributes;

  GrapplerItem item;
  GraphDef *graph = &item.graph;
  NodeDef *start_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(0, graph, &start_node));
  NodeDef *stop_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(10, graph, &stop_node));
  NodeDef *step_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(1, graph, &step_node));

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  NodeDef *range_node;
  TF_ASSERT_OK(graph_utils::AddNode("", "RangeDataset", range_inputs,
                                    empty_attributes, graph, &range_node));

  MapAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_TRUE(graph_utils::Compare(*graph, output));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
