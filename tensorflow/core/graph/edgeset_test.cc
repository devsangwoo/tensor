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

#include "tensorflow/core/graph/edgeset.h"

#include <set>
#include <vector>
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/test.h"
=======
#include "tensorflow/core/graph/edgeset.h"

#include "tensorflow/core/graph/graph.h"
#include <gtest/gtest.h>
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {
class EdgeSetTest : public ::testing::Test {
 public:
<<<<<<< HEAD
  EdgeSetTest() : edges_(nullptr) {}
  ~EdgeSetTest() override { delete[] edges_; }

  void MakeEdgeSet(int n) {
    if (edges_) {
      delete[] edges_;
    }
    edges_ = new Edge[n];
    eset_.clear();
    model_.clear();
    for (int i = 0; i < n; i++) {
      eset_.insert(&edges_[i]);
=======
  EdgeSetTest() : edges_(nullptr), eset_(nullptr) {}

  ~EdgeSetTest() override {
    delete eset_;
    delete[] edges_;
  }

  void MakeEdgeSet(int n) {
    delete eset_;
    delete[] edges_;
    edges_ = new Edge[n];
    eset_ = new EdgeSet;
    model_.clear();
    for (int i = 0; i < n; i++) {
      eset_->insert(&edges_[i]);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      model_.insert(&edges_[i]);
    }
  }

  void CheckSame() {
<<<<<<< HEAD
    EXPECT_EQ(model_.size(), eset_.size());
    EXPECT_EQ(model_.empty(), eset_.empty());
    std::vector<const Edge*> modelv(model_.begin(), model_.end());
    std::vector<const Edge*> esetv(eset_.begin(), eset_.end());
=======
    EXPECT_EQ(model_.size(), eset_->size());
    EXPECT_EQ(model_.empty(), eset_->empty());
    std::vector<const Edge*> modelv(model_.begin(), model_.end());
    std::vector<const Edge*> esetv(eset_->begin(), eset_->end());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    std::sort(modelv.begin(), modelv.end());
    std::sort(esetv.begin(), esetv.end());
    EXPECT_EQ(modelv.size(), esetv.size());
    for (size_t i = 0; i < modelv.size(); i++) {
      EXPECT_EQ(modelv[i], esetv[i]) << i;
    }
  }

<<<<<<< HEAD
  static constexpr int kInline = 64 / sizeof(const void*);
  Edge nonexistent_;
  Edge* edges_;
  EdgeSet eset_;
=======
  Edge nonexistent_;
  Edge* edges_;
  EdgeSet* eset_;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  std::set<const Edge*> model_;
};

namespace {

TEST_F(EdgeSetTest, Ops) {
<<<<<<< HEAD
  for (int n : {0, 1, 2, kInline + 1}) {
    MakeEdgeSet(n);
    CheckSame();
    EXPECT_EQ((n == 0), eset_.empty());
    EXPECT_EQ(n, eset_.size());

    eset_.clear();
    model_.clear();
    CheckSame();

    eset_.insert(&edges_[0]);
=======
  for (int n : {0, 1, 2, 3, 4, 10}) {
    MakeEdgeSet(n);
    CheckSame();
    EXPECT_EQ((n == 0), eset_->empty());
    EXPECT_EQ(n, eset_->size());

    eset_->clear();
    model_.clear();
    CheckSame();

    eset_->insert(&edges_[0]);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    model_.insert(&edges_[0]);
    CheckSame();
  }
}

// Try insert/erase of existing elements at different positions.
TEST_F(EdgeSetTest, Exists) {
<<<<<<< HEAD
  for (int n : {0, 1, 2, kInline + 1}) {
    MakeEdgeSet(n);
    for (int pos = 0; pos < n; pos++) {
      auto p = eset_.insert(&edges_[pos]);
      EXPECT_FALSE(p.second);
      EXPECT_EQ(&edges_[pos], *p.first);

      EXPECT_EQ(1, eset_.erase(&edges_[pos]));
=======
  for (int n : {0, 1, 2, 3, 4, 10}) {
    MakeEdgeSet(n);
    for (int pos = 0; pos < n; pos++) {
      MakeEdgeSet(n);
      auto p = eset_->insert(&edges_[pos]);
      EXPECT_FALSE(p.second);
      EXPECT_EQ(&edges_[pos], *p.first);

      EXPECT_EQ(1, eset_->erase(&edges_[pos]));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      model_.erase(&edges_[pos]);
      CheckSame();
    }
  }
}

// Try insert/erase of non-existent element.
TEST_F(EdgeSetTest, DoesNotExist) {
<<<<<<< HEAD
  for (int n : {0, 1, 2, kInline + 1}) {
    MakeEdgeSet(n);
    EXPECT_EQ(0, eset_.erase(&nonexistent_));
    auto p = eset_.insert(&nonexistent_);
=======
  for (int n : {0, 1, 2, 3, 4, 10}) {
    MakeEdgeSet(n);
    EXPECT_EQ(0, eset_->erase(&nonexistent_));
    auto p = eset_->insert(&nonexistent_);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    EXPECT_TRUE(p.second);
    EXPECT_EQ(&nonexistent_, *p.first);
  }
}

}  // namespace
}  // namespace tensorflow
