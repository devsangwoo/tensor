# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Smoke test for reading records from GCS to TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.python.lib.io import file_io

flags = tf.app.flags
flags.DEFINE_string("gcs_bucket_url", "",
                    "The URL to the GCS bucket in which the temporary "
                    "tfrecord file is to be written and read, e.g., "
                    "gs://my-gcs-bucket/test-directory")
flags.DEFINE_integer("num_examples", 10, "Number of examples to generate")

FLAGS = flags.FLAGS

def create_examples(num_examples, input_mean):
  """Create ExampleProto's containing data."""
  ids = np.arange(num_examples).reshape([num_examples, 1])
  inputs = np.random.randn(num_examples, 1) + input_mean
  target = inputs - input_mean
  examples = []
  for row in range(num_examples):
    ex = example_pb2.Example()
    ex.features.feature["id"].bytes_list.value.append(str(ids[row, 0]))
    ex.features.feature["target"].float_list.value.append(target[row, 0])
    ex.features.feature["inputs"].float_list.value.append(inputs[row, 0])
    examples.append(ex)
  return examples

def create_dir_test():
  """Verifies file_io directory handling methods ."""

  starttime = int(round(time.time() * 1000))
  dir_name = "%s/tf_gcs_test_%s" % (FLAGS.gcs_bucket_url, starttime)
  print("Creating dir %s" % dir_name)
  file_io.create_dir(dir_name)
  elapsed = int(round(time.time() * 1000)) - starttime
  print("Created directory in: %d milliseconds" % elapsed)
  # Check that the directory exists.
  dir_exists = file_io.is_directory(dir_name)
  print("%s directory exists: %s" % (dir_name, dir_exists))

  # List contents of just created directory.
  print("Listing directory %s." % dir_name)
  starttime = int(round(time.time() * 1000))
  print(file_io.list_directory(dir_name))
  elapsed = int(round(time.time() * 1000)) - starttime
  print("Listed directory %s in %s milliseconds" % (dir_name, elapsed))

  # Delete directory.
  print("Deleting directory %s." % dir_name)
  starttime = int(round(time.time() * 1000))
  file_io.delete_recursively(dir_name)
  elapsed = int(round(time.time() * 1000)) - starttime
  print("Deleted directory %s in %s milliseconds" % (dir_name, elapsed))

def create_object_test():
  """Verifies file_io's object manipulation methods ."""
  starttime = int(round(time.time() * 1000))
  dir_name = "%s/tf_gcs_test_%s" % (FLAGS.gcs_bucket_url, starttime)
  print("Creating dir %s." % dir_name)
  file_io.create_dir(dir_name)

  # Create a file in this directory.
  file_name = "%s/test_file.txt" % dir_name
  print("Creating file %s." % file_name)
  file_io.write_string_to_file(file_name, "test file creation.")

  list_files_pattern = "%s/test_file*.txt" % dir_name
  print("Getting files matching pattern %s." % list_files_pattern)
  files_list = file_io.get_matching_files(list_files_pattern)
  print(files_list)

  assert len(files_list) == 1
  assert files_list[0] == file_name

  # Cleanup test files.
  print("Deleting file %s." % file_name)
  file_io.delete_file(file_name)

  # Delete directory.
  print("Deleting directory %s." % dir_name)
  file_io.delete_recursively(dir_name)


if __name__ == "__main__":
  # Sanity check on the GCS bucket URL.
  if not FLAGS.gcs_bucket_url or not FLAGS.gcs_bucket_url.startswith("gs://"):
    print("ERROR: Invalid GCS bucket URL: \"%s\"" % FLAGS.gcs_bucket_url)
    sys.exit(1)

  # Generate random tfrecord path name.
  input_path = FLAGS.gcs_bucket_url + "/"
  input_path += "".join(random.choice("0123456789ABCDEF") for i in range(8))
  input_path += ".tfrecord"
  print("Using input path: %s" % input_path)

  # Verify that writing to the records file in GCS works.
  print("\n=== Testing writing and reading of GCS record file... ===")
  example_data = create_examples(FLAGS.num_examples, 5)
  with tf.python_io.TFRecordWriter(input_path) as hf:
    for e in example_data:
      hf.write(e.SerializeToString())

    print("Data written to: %s" % input_path)

  # Verify that reading from the tfrecord file works and that
  # tf_record_iterator works.
  record_iter = tf.python_io.tf_record_iterator(input_path)
  read_count = 0
  for r in record_iter:
    read_count += 1
  print("Read %d records using tf_record_iterator" % read_count)

  if read_count != FLAGS.num_examples:
    print("FAIL: The number of records read from tf_record_iterator (%d) "
          "differs from the expected number (%d)" % (read_count,
                                                     FLAGS.num_examples))
    sys.exit(1)

  # Verify that running the read op in a session works.
  print("\n=== Testing TFRecordReader.read op in a session... ===")
  with tf.Graph().as_default() as g:
    filename_queue = tf.train.string_input_producer([input_path], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      tf.train.start_queue_runners()
      index = 0
      for _ in range(FLAGS.num_examples):
        print("Read record: %d" % index)
        sess.run(serialized_example)
        index += 1

      # Reading one more record should trigger an exception.
      try:
        sess.run(serialized_example)
        print("FAIL: Failed to catch the expected OutOfRangeError while "
              "reading one more record than is available")
        sys.exit(1)
      except tf.errors.OutOfRangeError:
        print("Successfully caught the expected OutOfRangeError while "
              "reading one more record than is available")

  create_dir_test()
  create_object_test()
