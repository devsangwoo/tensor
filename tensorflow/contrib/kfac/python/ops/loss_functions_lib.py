# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Loss functions to be used by LayerCollection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,wildcard-import
from tensorflow.contrib.kfac.python.ops.loss_functions import *
from tensorflow.python.util.all_util import remove_undocumented
# pylint: enable=unused-import,line-too-long,wildcard-import

_allowed_symbols = [
    "LossFunction",
    "NegativeLogProbLoss",
    "NaturalParamsNegativeLogProbLoss",
    "DistributionNegativeLogProbLoss",
    "NormalMeanNegativeLogProbLoss",
    "CategoricalLogitsNegativeLogProbLoss",
    "MultiBernoulliNegativeLogProbLoss",
    "MultiBernoulliNegativeLogProbLoss",
    "insert_slice_in_zeros",
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
