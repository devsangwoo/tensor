# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.ragged in eager execution mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

from absl.testing import parameterized

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


class RaggedTensorTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(pylist=[[b'a', b'b'], [b'c']]),
      dict(pylist=[[[1, 2], [3]], [[4, 5, 6], [], [7]]]),
      dict(pylist=[[[1, 2], [3, 4]], [[5, 6], [], [7, 8]]], ragged_rank=1),
  ])
  def testRaggedTensorToList(self, pylist, ragged_rank=None):
    rt = ragged.constant(pylist, ragged_rank)
    self.assertEqual(rt.tolist(), pylist)

  expected = "RaggedTensor([['a', 'b'], ['c']])"
  if sys.version_info[0] == 3:
    expected = "RaggedTensor([[b'a', b'b'], [b'c']])"

  @parameterized.parameters([
      dict(pylist=[['a', 'b'], ['c']],
           expected=expected),
      dict(pylist=[[[1, 2], [3]], [[4, 5, 6], [], [7]]],
           expected='RaggedTensor([[[1, 2], [3]], [[4, 5, 6], [], [7]]])'),
  ])
  def testRaggedTensorStr(self, pylist, expected):
    rt = ragged.constant(pylist)
    self.assertEqual(str(rt), expected)


if __name__ == '__main__':
  ops.enable_eager_execution()
  googletest.main()
