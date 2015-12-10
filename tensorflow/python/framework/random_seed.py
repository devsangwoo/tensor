# Copyright 2015 Google Inc. All Rights Reserved.
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

"""For seeding individual ops based on a graph-level seed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops


_DEFAULT_GRAPH_SEED = 87654321


def get_seed(op_seed):
  """Returns the local seeds an operation should use given an op-specific seed.

  Given operation-specific seed, `op_seed`, this helper function returns two
  seeds derived from graph-level and op-level seeds. Many random operations
  internally use the two seeds to allow user to change the seed globally for a
  graph, or for only specific operations.

  For details on how the graph-level seed interacts with op seeds, see
  [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed).

  Args:
    op_seed: integer.

  Returns:
    A tuple of two integers that should be used for the local seed of this
    operation.
  """
  graph_seed = ops.get_default_graph().seed
  if graph_seed is not None:
    if op_seed is not None:
      return graph_seed, op_seed
    else:
      return graph_seed, ops.get_default_graph()._last_id
  else:
    if op_seed is not None:
      return _DEFAULT_GRAPH_SEED, op_seed
    else:
      return None, None


def set_random_seed(seed):
  """Sets the graph-level random seed.

  Operations that rely on a random seed actually derive it from two seeds:
  the graph-level and operation-level seeds. This sets the graph-level seed.

  Its interactions with operation-level seeds is as follows:

    1. If neither the graph-level nor the operation seed is set:
      A random seed is used for this op.
    2. If the graph-level seed is set, but the operation seed is not:
      The system deterministically picks an operation seed in conjunction
      with the graph-level seed so that it gets a unique random sequence.
    3. If the graph-level seed is not set, but the operation seed is set:
      A default graph-level seed and the specified operation seed are used to
      determine the random sequence.
    4. If both the graph-level and the operation seed are set:
      Both seeds are used in conjunction to determine the random sequence.

  To illustrate the user-visible effects, consider these examples:

  To generate different sequences across sessions, set neither
  graph-level nor op-level seeds:

  ```python
  a = tf.random_uniform([1])
  b = tf.random_normal([1])

  print("Session 1")
  with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

  print("Session 2")
  with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A3'
    print(sess2.run(a))  # generates 'A4'
    print(sess2.run(b))  # generates 'B3'
    print(sess2.run(b))  # generates 'B4'
  ```

  To generate the same repeatable sequence for an op across sessions, set the
  seed for the op:

  ```python
  a = tf.random_uniform([1], seed=1)
  b = tf.random_normal([1])

  # Repeatedly running this block with the same graph will generate the same
  # sequence of values for 'a', but different sequences of values for 'b'.
  print("Session 1")
  with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

  print("Session 2")
  with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B3'
    print(sess2.run(b))  # generates 'B4'
  ```

  To make the random sequences generated by all ops be repeatable across
  sessions, set a graph-level seed:

  ```python
  tf.set_random_seed(1234)
  a = tf.random_uniform([1])
  b = tf.random_normal([1])

  # Repeatedly running this block with the same graph will generate different
  # sequences of 'a' and 'b'.
  print("Session 1")
  with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

  print("Session 2")
  with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B1'
    print(sess2.run(b))  # generates 'B2'
  ```

  Args:
    seed: integer.
  """
  ops.get_default_graph().seed = seed
