# Copyright 2016 Google Inc. All Rights Reserved.
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
"""TFGAN grouped API. Please see README.md for details and usage."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Collapse TFGAN into a tiered namespace.
from tensorflow.contrib.gan.python import features
from tensorflow.contrib.gan.python import losses

del absolute_import
del division
del print_function
