# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for IgniteDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.ignite import IgniteDataset
from tensorflow.python.framework import errors
from tensorflow.python.platform import test

class IgniteDatasetTest(test.TestCase):
  """The Apache Ignite servers have to setup before the test and tear down
     after the test manually. The docker engine has to be installed.

     To setup Apache Ignite servers:
     $ bash start_ignite.sh

     To tear down Apache Ignite servers:
     $ bash stop_ignite.sh
  """

  def test_ignite_dataset_with_plain_client(self):
    """Test Ignite Dataset with plain client.
    """
    self._clear_env()
    ds = IgniteDataset(cache_name="SQL_PUBLIC_TEST_CACHE", port=42300)
    self._check_dataset(ds)

  def test_ignite_dataset_with_ssl_client(self):
    """Test Ignite Dataset with ssl client.
    """
    self._clear_env()
    os.environ["IGNITE_DATASET_CERTFILE"] = os.path.dirname(
        os.path.realpath(__file__)) + "/keystore/client.pem"
    os.environ["IGNITE_DATASET_CERT_PASSWORD"] = "123456"

    ds = IgniteDataset(cache_name="SQL_PUBLIC_TEST_CACHE", port=42301,
                       certfile=os.environ["IGNITE_DATASET_CERTFILE"],
                       cert_password=os.environ["IGNITE_DATASET_CERT_PASSWORD"])
    self._check_dataset(ds)

  def test_ignite_dataset_with_ssl_client_and_auth(self):
    """Test Ignite Dataset with ssl client and authentication.
    """
    self._clear_env()
    os.environ['IGNITE_DATASET_USERNAME'] = "ignite"
    os.environ['IGNITE_DATASET_PASSWORD'] = "ignite"
    os.environ['IGNITE_DATASET_CERTFILE'] = os.path.dirname(
        os.path.realpath(__file__)) + "/keystore/client.pem"
    os.environ['IGNITE_DATASET_CERT_PASSWORD'] = "123456"

    ds = IgniteDataset(cache_name="SQL_PUBLIC_TEST_CACHE", port=42302,
                       certfile=os.environ['IGNITE_DATASET_CERTFILE'],
                       cert_password=os.environ['IGNITE_DATASET_CERT_PASSWORD'],
                       username=os.environ['IGNITE_DATASET_USERNAME'],
                       password=os.environ['IGNITE_DATASET_PASSWORD'])
    self._check_dataset(ds)

  def _clear_env(self):
    """Clears environment variables used by Ignite Dataset.
    """
    if 'IGNITE_DATASET_USERNAME' in os.environ:
      del os.environ['IGNITE_DATASET_USERNAME']
    if 'IGNITE_DATASET_PASSWORD' in os.environ:
      del os.environ['IGNITE_DATASET_PASSWORD']
    if 'IGNITE_DATASET_CERTFILE' in os.environ:
      del os.environ['IGNITE_DATASET_CERTFILE']
    if 'IGNITE_DATASET_CERT_PASSWORD' in os.environ:
      del os.environ['IGNITE_DATASET_CERT_PASSWORD']

  def _check_dataset(self, dataset):
    """Checks that dataset provids correct data.
    """
    self.assertEqual(tf.int64, dataset.output_types['key'])
    self.assertEqual(tf.string, dataset.output_types['val']['NAME'])
    self.assertEqual(tf.int64, dataset.output_types['val']['VAL'])

    it = dataset.make_one_shot_iterator()
    ne = it.get_next()

    with tf.Session() as sess:
      rows = [sess.run(ne), sess.run(ne), sess.run(ne)]
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(ne)

    self.assertEqual({'key': 1, 'val': {'NAME': b'TEST1', 'VAL': 42}},\
      rows[0])
    self.assertEqual({'key': 2, 'val': {'NAME': b'TEST2', 'VAL': 43}},\
      rows[1])
    self.assertEqual({'key': 3, 'val': {'NAME': b'TEST3', 'VAL': 44}},\
      rows[2])

if __name__ == "__main__":
  test.main()
