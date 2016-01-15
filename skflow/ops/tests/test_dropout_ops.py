#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import tensorflow as tf

from skflow import ops


class DropoutTest(tf.test.TestCase):

    def test_dropout_float(self):
        with self.test_session():
            x = tf.placeholder(tf.float32, [5, 5])
            y = ops.dropout(x, 0.5)
            probs = tf.get_collection(ops.DROPOUTS)
            self.assertEqual(len(probs), 1)


if __name__ == '__main__':
    tf.test.main()
