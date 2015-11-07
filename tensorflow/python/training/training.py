<<<<<<< HEAD
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Support for training models.

See the [Training](https://tensorflow.org/api_guides/python/train) guide.
"""

# Optimizers.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order,unused-import
from tensorflow.python.ops.sdca_ops import sdca_optimizer
from tensorflow.python.ops.sdca_ops import sdca_fprint
from tensorflow.python.ops.sdca_ops import sdca_shrink_l1
from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.adagrad_da import AdagradDAOptimizer
from tensorflow.python.training.proximal_adagrad import ProximalAdagradOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.experimental.loss_scale_optimizer import MixedPrecisionLossScaleOptimizer
from tensorflow.python.training.experimental.mixed_precision import enable_mixed_precision_graph_rewrite
from tensorflow.python.training.experimental.mixed_precision import enable_mixed_precision_graph_rewrite_v1
=======
# pylint: disable=wildcard-import,unused-import,g-bad-import-order,line-too-long
"""This library provides a set of classes and functions that helps train models.

## Optimizers.

The Optimizer base class provides methods to compute gradients for a loss and
apply gradients to variables.  A collection of subclasses implement classic
optimization algorithms such as GradientDescent and Adagrad.

You never instantiate the Optimizer class itself, but instead instantiate one
of the subclasses.

@@Optimizer

@@GradientDescentOptimizer
@@AdagradOptimizer
@@MomentumOptimizer
@@AdamOptimizer
@@FtrlOptimizer
@@RMSPropOptimizer

## Gradient Computation.

TensorFlow provides functions to compute the derivatives for a given
TensorFlow computation graph, adding operations to the graph. The
optimizer classes automatically compute derivatives on your graph, but
creators of new Optimizers or expert users can call the lower-level
functions below.

@@gradients
@@AggregationMethod

@@stop_gradient


## Gradient Clipping

TensorFlow provides several operations that you can use to add clipping
functions to your graph. You can use these functions to perform general data
clipping, but they're particularly useful for handling exploding or vanishing
gradients.

@@clip_by_value
@@clip_by_norm
@@clip_by_average_norm
@@clip_by_global_norm
@@global_norm

## Decaying the learning rate.
@@exponential_decay

## Moving Averages.

Some training algorithms, such as GradientDescent and Momentum often benefit
from maintaining a moving average of variables during optimization.  Using the
moving averages for evaluations often improve results significantly.

@@ExponentialMovingAverage

## Coordinator and QueueRunner.

See [Threading and Queues](../../how_tos/threading_and_queues/index.md)
for how to use threads and queues.  For documentation on the Queue API,
see [Queues](../../api_docs/python/io_ops.md#queues).

@@Coordinator
@@QueueRunner
@@add_queue_runner
@@start_queue_runners

## Summary Operations.

The following ops output
[`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
protocol buffers as serialized string tensors.

You can fetch the output of a summary op in a session, and pass it to a
[SummaryWriter](train.md#SummaryWriter) to append it to an event file.  You can
then use TensorBoard to visualize the contents of the event files.  See
[TensorBoard and Summaries](../../how_tos/summaries_and_tensorboard/index.md)
for more details.

@@scalar_summary
@@image_summary
@@histogram_summary
@@zero_fraction

@@merge_summary
@@merge_all_summaries

## Adding Summaries to Event Files.

See [Summaries and
TensorBoard](../../how_tos/summaries_and_tensorboard/index.md) for an
overview of summaries, event files, and visualization in TensorBoard.

@@SummaryWriter
@@summary_iterator

## Training utilities.

@@global_step
@@write_graph
"""

# Optimizers.
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.moving_averages import ExponentialMovingAverage
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
<<<<<<< HEAD
from tensorflow.python.training.proximal_gradient_descent import ProximalGradientDescentOptimizer
from tensorflow.python.training.sync_replicas_optimizer import SyncReplicasOptimizer

# Utility classes for training.
from tensorflow.python.training.coordinator import Coordinator
from tensorflow.python.training.coordinator import LooperThread
# go/tf-wildcard-import
# pylint: disable=wildcard-import
=======

# Utility classes for training.
from tensorflow.python.training.coordinator import Coordinator
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
from tensorflow.python.training.queue_runner import *

# For the module level doc.
from tensorflow.python.training import input as _input
<<<<<<< HEAD
from tensorflow.python.training.input import *  # pylint: disable=redefined-builtin
# pylint: enable=wildcard-import

from tensorflow.python.training.basic_session_run_hooks import get_or_create_steps_per_run_variable
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.basic_session_run_hooks import LoggingTensorHook
from tensorflow.python.training.basic_session_run_hooks import StopAtStepHook
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverHook
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverListener
from tensorflow.python.training.basic_session_run_hooks import StepCounterHook
from tensorflow.python.training.basic_session_run_hooks import NanLossDuringTrainingError
from tensorflow.python.training.basic_session_run_hooks import NanTensorHook
from tensorflow.python.training.basic_session_run_hooks import SummarySaverHook
from tensorflow.python.training.basic_session_run_hooks import GlobalStepWaiterHook
from tensorflow.python.training.basic_session_run_hooks import FinalOpsHook
from tensorflow.python.training.basic_session_run_hooks import FeedFnHook
from tensorflow.python.training.basic_session_run_hooks import ProfilerHook
from tensorflow.python.training.basic_loops import basic_train_loop
from tensorflow.python.training.tracking.python_state import PythonState
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.python.training.checkpoint_utils import init_from_checkpoint
from tensorflow.python.training.checkpoint_utils import list_variables
from tensorflow.python.training.checkpoint_utils import load_checkpoint
from tensorflow.python.training.checkpoint_utils import load_variable

from tensorflow.python.training.device_setter import replica_device_setter
from tensorflow.python.training.monitored_session import Scaffold
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
from tensorflow.python.training.monitored_session import SessionCreator
from tensorflow.python.training.monitored_session import ChiefSessionCreator
from tensorflow.python.training.monitored_session import WorkerSessionCreator
from tensorflow.python.training.monitored_session import MonitoredSession
from tensorflow.python.training.monitored_session import SingularMonitoredSession
from tensorflow.python.training.saver import Saver
from tensorflow.python.training.checkpoint_management import checkpoint_exists
from tensorflow.python.training.checkpoint_management import generate_checkpoint_state_proto
from tensorflow.python.training.checkpoint_management import get_checkpoint_mtimes
from tensorflow.python.training.checkpoint_management import get_checkpoint_state
from tensorflow.python.training.checkpoint_management import latest_checkpoint
from tensorflow.python.training.checkpoint_management import update_checkpoint_state
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.training.saver import import_meta_graph
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.session_run_hook import SessionRunContext
from tensorflow.python.training.session_run_hook import SessionRunValues
from tensorflow.python.training.session_manager import SessionManager
from tensorflow.python.training.summary_io import summary_iterator
from tensorflow.python.training.supervisor import Supervisor
from tensorflow.python.training.training_util import write_graph
from tensorflow.python.training.training_util import global_step
from tensorflow.python.training.training_util import get_global_step
from tensorflow.python.training.training_util import assert_global_step
from tensorflow.python.training.training_util import create_global_step
from tensorflow.python.training.training_util import get_or_create_global_step
from tensorflow.python.training.warm_starting_util import VocabInfo
from tensorflow.python.training.warm_starting_util import warm_start
from tensorflow.python.training.py_checkpoint_reader import NewCheckpointReader
from tensorflow.python.util.tf_export import tf_export

# pylint: disable=wildcard-import
# Training data protos.
from tensorflow.core.example.example_pb2 import *
from tensorflow.core.example.feature_pb2 import *
from tensorflow.core.protobuf.saver_pb2 import *

# Utility op.  Open Source. TODO(touts): move to nn?
from tensorflow.python.training.learning_rate_decay import *
# pylint: enable=wildcard-import

# Distributed computing support.
from tensorflow.core.protobuf.cluster_pb2 import ClusterDef
from tensorflow.core.protobuf.cluster_pb2 import JobDef
from tensorflow.core.protobuf.tensorflow_server_pb2 import ServerDef
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.training.server_lib import Server

# pylint: disable=undefined-variable
tf_export("train.BytesList")(BytesList)
tf_export("train.ClusterDef")(ClusterDef)
tf_export("train.Example")(Example)
tf_export("train.Feature")(Feature)
tf_export("train.Features")(Features)
tf_export("train.FeatureList")(FeatureList)
tf_export("train.FeatureLists")(FeatureLists)
tf_export("train.FloatList")(FloatList)
tf_export("train.Int64List")(Int64List)
tf_export("train.JobDef")(JobDef)
tf_export(v1=["train.SaverDef"])(SaverDef)
tf_export("train.SequenceExample")(SequenceExample)
tf_export("train.ServerDef")(ServerDef)
# pylint: enable=undefined-variable
=======
from tensorflow.python.training.input import *

from tensorflow.python.training.saver import get_checkpoint_state
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.python.training.saver import Saver
from tensorflow.python.training.saver import update_checkpoint_state
from tensorflow.python.training.summary_io import summary_iterator
from tensorflow.python.training.summary_io import SummaryWriter
from tensorflow.python.training.training_util import write_graph
from tensorflow.python.training.training_util import global_step

# Training data protos.
from tensorflow.core.example.example_pb2 import *
from tensorflow.core.example.feature_pb2 import *

# Utility op.  Open Source. TODO(mdevin): move to nn?
from tensorflow.python.training.learning_rate_decay import exponential_decay
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
