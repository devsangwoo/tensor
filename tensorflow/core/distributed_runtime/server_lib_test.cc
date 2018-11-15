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

#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ServerLibTest, NewServerNoFactoriesAccept) {
  ServerDef server_def;
  server_def.set_protocol("fake_protocol");
  std::unique_ptr<ServerInterface> server;
  Status s = NewServer(server_def, &server);
  ASSERT_NE(s, Status::OK());
  EXPECT_TRUE(str_util::StrContains(
      s.error_message(),
      "No server factory registered for the given ServerDef"));
  EXPECT_TRUE(str_util::StrContains(s.error_message(),
                                    "The available server factories are: ["));
}

}  // namespace tensorflow
