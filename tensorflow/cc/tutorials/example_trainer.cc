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
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

using tensorflow::string;
using tensorflow::int32;
=======
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/command_line_flags.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/tensor.h"
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

namespace tensorflow {
namespace example {

struct Options {
<<<<<<< HEAD
  int num_concurrent_sessions = 1;   // The number of concurrent sessions
=======
  int num_concurrent_sessions = 10;  // The number of concurrent sessions
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  int num_concurrent_steps = 10;     // The number of concurrent steps
  int num_iterations = 100;          // Each step repeats this many times
  bool use_gpu = false;              // Whether to use gpu in the training
};

<<<<<<< HEAD
=======
TF_DEFINE_int32(num_concurrent_sessions, 10, "Number of concurrent sessions");
TF_DEFINE_int32(num_concurrent_steps, 10, "Number of concurrent steps");
TF_DEFINE_int32(num_iterations, 100, "Number of iterations");
TF_DEFINE_bool(use_gpu, false, "Whether to use gpu in the training");

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
// A = [3 2; -1 0]; x = rand(2, 1);
// We want to compute the largest eigenvalue for A.
// repeat x = y / y.norm(); y = A * x; end
GraphDef CreateGraphDef() {
  // TODO(jeff,opensource): This should really be a more interesting
  // computation.  Maybe turn this into an mnist model instead?
<<<<<<< HEAD
  Scope root = Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  // A = [3 2; -1 0].  Using Const<float> means the result will be a
  // float tensor even though the initializer has integers.
  auto a = Const<float>(root, {{3, 2}, {-1, 0}});

  // x = [1.0; 1.0]
  auto x = Const(root.WithOpName("x"), {{1.f}, {1.f}});

  // y = A * x
  auto y = MatMul(root.WithOpName("y"), a, x);

  // y2 = y.^2
  auto y2 = Square(root, y);

  // y2_sum = sum(y2).  Note that you can pass constants directly as
  // inputs.  Sum() will automatically create a Const node to hold the
  // 0 value.
  auto y2_sum = Sum(root, y2, 0);

  // y_norm = sqrt(y2_sum)
  auto y_norm = Sqrt(root, y2_sum);

  // y_normalized = y ./ y_norm
  Div(root.WithOpName("y_normalized"), y, y_norm);

  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));

=======
  GraphDefBuilder b;
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  // Store rows [3, 2] and [-1, 0] in row major format.
  Node* a = Const({3.f, 2.f, -1.f, 0.f}, {2, 2}, b.opts());

  // x is from the feed.
  Node* x = Const({0.f}, {2, 1}, b.opts().WithName("x"));

  // y = A * x
  Node* y = MatMul(a, x, b.opts().WithName("y"));

  // y2 = y.^2
  Node* y2 = Square(y, b.opts());

  // y2_sum = sum(y2)
  Node* y2_sum = Sum(y2, Const(0, b.opts()), b.opts());

  // y_norm = sqrt(y2_sum)
  Node* y_norm = Sqrt(y2_sum, b.opts());

  // y_normalized = y ./ y_norm
  Div(y, y_norm, b.opts().WithName("y_normalized"));

  GraphDef def;
  TF_CHECK_OK(b.ToGraphDef(&def));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return def;
}

string DebugString(const Tensor& x, const Tensor& y) {
  CHECK_EQ(x.NumElements(), 2);
  CHECK_EQ(y.NumElements(), 2);
  auto x_flat = x.flat<float>();
  auto y_flat = y.flat<float>();
<<<<<<< HEAD
  // Compute an estimate of the eigenvalue via
  //      (x' A x) / (x' x) = (x' y) / (x' x)
  // and exploit the fact that x' x = 1 by assumption
  Eigen::Tensor<float, 0, Eigen::RowMajor> lambda = (x_flat * y_flat).sum();
  return strings::Printf("lambda = %8.6f x = [%8.6f %8.6f] y = [%8.6f %8.6f]",
                         lambda(), x_flat(0), x_flat(1), y_flat(0), y_flat(1));
=======
  const float lambda = y_flat(0) / x_flat(0);
  return strings::Printf("lambda = %8.6f x = [%8.6f %8.6f] y = [%8.6f %8.6f]",
                         lambda, x_flat(0), x_flat(1), y_flat(0), y_flat(1));
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

void ConcurrentSteps(const Options* opts, int session_index) {
  // Creates a session.
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  GraphDef def = CreateGraphDef();
  if (options.target.empty()) {
<<<<<<< HEAD
    graph::SetDefaultDevice(opts->use_gpu ? "/device:GPU:0" : "/cpu:0", &def);
=======
    graph::SetDefaultDevice(opts->use_gpu ? "/gpu:0" : "/cpu:0", &def);
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  }

  TF_CHECK_OK(session->Create(def));

  // Spawn M threads for M concurrent steps.
  const int M = opts->num_concurrent_steps;
<<<<<<< HEAD
  std::unique_ptr<thread::ThreadPool> step_threads(
      new thread::ThreadPool(Env::Default(), "trainer", M));

  for (int step = 0; step < M; ++step) {
    step_threads->Schedule([&session, opts, session_index, step]() {
      // Randomly initialize the input.
      Tensor x(DT_FLOAT, TensorShape({2, 1}));
      auto x_flat = x.flat<float>();
      x_flat.setRandom();
      Eigen::Tensor<float, 0, Eigen::RowMajor> inv_norm =
          x_flat.square().sum().sqrt().inverse();
      x_flat = x_flat * inv_norm();
=======
  thread::ThreadPool step_threads(Env::Default(), "trainer", M);

  for (int step = 0; step < M; ++step) {
    step_threads.Schedule([&session, opts, session_index, step]() {
      // Randomly initialize the input.
      Tensor x(DT_FLOAT, TensorShape({2, 1}));
      x.flat<float>().setRandom();
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

      // Iterations.
      std::vector<Tensor> outputs;
      for (int iter = 0; iter < opts->num_iterations; ++iter) {
        outputs.clear();
        TF_CHECK_OK(
            session->Run({{"x", x}}, {"y:0", "y_normalized:0"}, {}, &outputs));
<<<<<<< HEAD
        CHECK_EQ(size_t{2}, outputs.size());
=======
        CHECK_EQ(2, outputs.size());
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

        const Tensor& y = outputs[0];
        const Tensor& y_norm = outputs[1];
        // Print out lambda, x, and y.
        std::printf("%06d/%06d %s\n", session_index, step,
                    DebugString(x, y).c_str());
        // Copies y_normalized to x.
        x = y_norm;
      }
    });
  }

<<<<<<< HEAD
  // Delete the threadpool, thus waiting for all threads to complete.
  step_threads.reset(nullptr);
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  TF_CHECK_OK(session->Close());
}

void ConcurrentSessions(const Options& opts) {
  // Spawn N threads for N concurrent sessions.
  const int N = opts.num_concurrent_sessions;
<<<<<<< HEAD

  // At the moment our Session implementation only allows
  // one concurrently computing Session on GPU.
  CHECK_EQ(1, N) << "Currently can only have one concurrent session.";

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  thread::ThreadPool session_threads(Env::Default(), "trainer", N);
  for (int i = 0; i < N; ++i) {
    session_threads.Schedule(std::bind(&ConcurrentSteps, &opts, i));
  }
}

}  // end namespace example
}  // end namespace tensorflow

<<<<<<< HEAD
namespace {

bool ParseInt32Flag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    int32* dst) {
  if (absl::ConsumePrefix(&arg, flag) && absl::ConsumePrefix(&arg, "=")) {
    char extra;
    return (sscanf(arg.data(), "%d%c", dst, &extra) == 1);
  }

  return false;
}

bool ParseBoolFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                   bool* dst) {
  if (absl::ConsumePrefix(&arg, flag)) {
    if (arg.empty()) {
      *dst = true;
      return true;
    }

    if (arg == "=true") {
      *dst = true;
      return true;
    } else if (arg == "=false") {
      *dst = false;
      return true;
    }
  }

  return false;
}

}  // namespace

int main(int argc, char* argv[]) {
  tensorflow::example::Options opts;
  std::vector<char*> unknown_flags;
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--") {
      while (i < argc) {
        unknown_flags.push_back(argv[i]);
        ++i;
      }
      break;
    }

    if (ParseInt32Flag(argv[i], "--num_concurrent_sessions",
                       &opts.num_concurrent_sessions) ||
        ParseInt32Flag(argv[i], "--num_concurrent_steps",
                       &opts.num_concurrent_steps) ||
        ParseInt32Flag(argv[i], "--num_iterations", &opts.num_iterations) ||
        ParseBoolFlag(argv[i], "--use_gpu", &opts.use_gpu)) {
      continue;
    }

    fprintf(stderr, "Unknown flag: %s\n", argv[i]);
    return -1;
  }

  // Passthrough any unknown flags.
  int dst = 1;  // Skip argv[0]
  for (char* f : unknown_flags) {
    argv[dst++] = f;
  }
  argv[dst++] = nullptr;
  argc = static_cast<int>(unknown_flags.size() + 1);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
=======
int main(int argc, char* argv[]) {
  tensorflow::example::Options opts;
  tensorflow::Status s = tensorflow::ParseCommandLineFlags(&argc, argv);
  if (!s.ok()) {
    LOG(FATAL) << "Error parsing command line flags: " << s.ToString();
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  opts.num_concurrent_sessions =
      tensorflow::example::FLAGS_num_concurrent_sessions;
  opts.num_concurrent_steps = tensorflow::example::FLAGS_num_concurrent_steps;
  opts.num_iterations = tensorflow::example::FLAGS_num_iterations;
  opts.use_gpu = tensorflow::example::FLAGS_use_gpu;
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  tensorflow::example::ConcurrentSessions(opts);
}
