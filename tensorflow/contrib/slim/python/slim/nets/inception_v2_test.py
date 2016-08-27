# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for nets.inception_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


class InceptionV2Test(tf.test.TestCase):

  def testBuildClassificationNetwork(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, 3))
    logits, end_points = inception.inception_v2(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('InceptionV2/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertTrue('Predictions' in end_points)
    self.assertListEqual(end_points['Predictions'].get_shape().as_list(),
                         [batch_size, num_classes])

  def testBuildBaseNetwork(self):
    batch_size = 5
    height, width = 224, 224

    inputs = tf.random_uniform((batch_size, height, width, 3))
    mixed_5c, end_points = inception.inception_v2_base(inputs)
    self.assertTrue(mixed_5c.op.name.startswith('InceptionV2/Mixed_5c'))
    self.assertListEqual(mixed_5c.get_shape().as_list(),
                         [batch_size, 7, 7, 1024])
    expected_endpoints = ['Mixed_3b', 'Mixed_3c', 'Mixed_4a', 'Mixed_4b',
                          'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a',
                          'Mixed_5b', 'Mixed_5c', 'Conv2d_1a_7x7',
                          'MaxPool_2a_3x3', 'Conv2d_2b_1x1', 'Conv2d_2c_3x3',
                          'MaxPool_3a_3x3']
    self.assertItemsEqual(end_points.keys(), expected_endpoints)

  def testBuildOnlyUptoFinalEndpoint(self):
    batch_size = 5
    height, width = 224, 224
    endpoints = ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
                 'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
                 'Mixed_4a', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
                 'Mixed_5a', 'Mixed_5b', 'Mixed_5c']
    for index, endpoint in enumerate(endpoints):
      with tf.Graph().as_default():
        inputs = tf.random_uniform((batch_size, height, width, 3))
        out_tensor, end_points = inception.inception_v2_base(
            inputs, final_endpoint=endpoint)
        self.assertTrue(out_tensor.op.name.startswith(
            'InceptionV2/' + endpoint))
        self.assertItemsEqual(endpoints[:index+1], end_points)

  def testBuildAndCheckAllEndPointsUptoMixed5c(self):
    batch_size = 5
    height, width = 224, 224

    inputs = tf.random_uniform((batch_size, height, width, 3))
    _, end_points = inception.inception_v2_base(inputs,
                                                final_endpoint='Mixed_5c')
    endpoints_shapes = {'Mixed_3b': [batch_size, 28, 28, 256],
                        'Mixed_3c': [batch_size, 28, 28, 320],
                        'Mixed_4a': [batch_size, 14, 14, 576],
                        'Mixed_4b': [batch_size, 14, 14, 576],
                        'Mixed_4c': [batch_size, 14, 14, 576],
                        'Mixed_4d': [batch_size, 14, 14, 576],
                        'Mixed_4e': [batch_size, 14, 14, 576],
                        'Mixed_5a': [batch_size, 7, 7, 1024],
                        'Mixed_5b': [batch_size, 7, 7, 1024],
                        'Mixed_5c': [batch_size, 7, 7, 1024],
                        'Conv2d_1a_7x7': [batch_size, 112, 112, 64],
                        'MaxPool_2a_3x3': [batch_size, 56, 56, 64],
                        'Conv2d_2b_1x1': [batch_size, 56, 56, 64],
                        'Conv2d_2c_3x3': [batch_size, 56, 56, 192],
                        'MaxPool_3a_3x3': [batch_size, 28, 28, 192]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testModelHasExpectedNumberOfParameters(self):
    batch_size = 5
    height, width = 224, 224
    inputs = tf.random_uniform((batch_size, height, width, 3))
    with slim.arg_scope(inception.inception_v2_arg_scope()):
      inception.inception_v2_base(inputs)
    total_params, _ = slim.model_analyzer.analyze_vars(
        slim.get_model_variables())
    self.assertAlmostEqual(10173112, total_params)

  def testBuildEndPointsWithDepthMultiplierLessThanOne(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, 3))
    _, end_points = inception.inception_v2(inputs, num_classes)

    endpoint_keys = [key for key in end_points.keys()
                     if key.startswith('Mixed') or key.startswith('Conv')]

    _, end_points_with_multiplier = inception.inception_v2(
        inputs, num_classes, scope='depth_multiplied_net',
        depth_multiplier=0.5)

    for key in endpoint_keys:
      original_depth = end_points[key].get_shape().as_list()[3]
      new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
      self.assertEqual(0.5 * original_depth, new_depth)

  def testBuildEndPointsWithDepthMultiplierGreaterThanOne(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, 3))
    _, end_points = inception.inception_v2(inputs, num_classes)

    endpoint_keys = [key for key in end_points.keys()
                     if key.startswith('Mixed') or key.startswith('Conv')]

    _, end_points_with_multiplier = inception.inception_v2(
        inputs, num_classes, scope='depth_multiplied_net',
        depth_multiplier=2.0)

    for key in endpoint_keys:
      original_depth = end_points[key].get_shape().as_list()[3]
      new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
      self.assertEqual(2.0 * original_depth, new_depth)

  def testRaiseValueErrorWithInvalidDepthMultiplier(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, 3))
    with self.assertRaises(ValueError):
      _ = inception.inception_v2(inputs, num_classes, depth_multiplier=-0.1)
    with self.assertRaises(ValueError):
      _ = inception.inception_v2(inputs, num_classes, depth_multiplier=0.0)

  def testHalfSizeImages(self):
    batch_size = 5
    height, width = 112, 112
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, 3))
    logits, end_points = inception.inception_v2(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('InceptionV2/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    pre_pool = end_points['Mixed_5c']
    self.assertListEqual(pre_pool.get_shape().as_list(),
                         [batch_size, 4, 4, 1024])

  def testUnknownImageShape(self):
    tf.reset_default_graph()
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000
    input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))
      logits, end_points = inception.inception_v2(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('InceptionV2/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['Mixed_5c']
      feed_dict = {inputs: input_np}
      tf.initialize_all_variables().run()
      pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
      self.assertListEqual(list(pre_pool_out.shape), [batch_size, 7, 7, 1024])

  def testUnknowBatchSize(self):
    batch_size = 1
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.placeholder(tf.float32, (None, height, width, 3))
    logits, _ = inception.inception_v2(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('InceptionV2/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [None, num_classes])
    images = tf.random_uniform((batch_size, height, width, 3))

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch_size, num_classes))

  def testEvaluation(self):
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000

    eval_inputs = tf.random_uniform((batch_size, height, width, 3))
    logits, _ = inception.inception_v2(eval_inputs, num_classes,
                                       is_training=False)
    predictions = tf.argmax(logits, 1)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (batch_size,))

  def testTrainEvalWithReuse(self):
    train_batch_size = 5
    eval_batch_size = 2
    height, width = 150, 150
    num_classes = 1000

    train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
    inception.inception_v2(train_inputs, num_classes)
    eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
    logits, _ = inception.inception_v2(eval_inputs, num_classes, reuse=True)
    predictions = tf.argmax(logits, 1)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (eval_batch_size,))

  def testLogitsNotSqueezed(self):
    num_classes = 25
    images = tf.random_uniform([1, 224, 224, 3])
    logits, _ = inception.inception_v2(images,
                                       num_classes=num_classes,
                                       spatial_squeeze=False)

    with self.test_session() as sess:
      tf.initialize_all_variables().run()
      logits_out = sess.run(logits)
      self.assertListEqual(list(logits_out.shape), [1, 1, 1, num_classes])


if __name__ == '__main__':
  tf.test.main()
