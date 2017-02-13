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
"""The Gamma distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops


__all__ = [
    "Gamma",
    "GammaWithSoftplusConcentrationRate",
]


class Gamma(distribution.Distribution):
  """Gamma distribution.

  The Gamma distribution is defined over positive real numbers using
  parameters `concentration` (aka "alpha") and `rate` (aka "beta").

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta, x > 0) = x**(alpha - 1) exp(-x beta) / Z
  Z = Gamma(alpha) beta**alpha
  ```

  where:

  * `concentration = alpha`, `alpha > 0`,
  * `rate = beta`, `beta > 0`,
  * `Z` is the normalizing constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The cumulative density function (cdf) is,

  ```none
  cdf(x; alpha, beta, x > 0) = GammaInc(alpha, beta x) / Gamma(alpha)
  ```

  where `GammaInc` is the [lower incomplete Gamma function](
  https://en.wikipedia.org/wiki/Incomplete_gamma_function).

  The parameters can be intuited via their relationship to mean and stddev,

  ```none
  concentration = alpha = (mean / stddev)**2
  rate = beta = mean / stddev**2 = concentration / mean
  ```

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  WARNING: This distribution may draw 0-valued samples for small `concentration`
  values. See note in `tf.random_gamma` docstring.

  #### Examples

  ```python
  dist = Gamma(concentration=3.0, rate=2.0)
  dist2 = Gamma(concentration=[3.0, 4.0], rate=[2.0, 3.0])
  ```

  """

  def __init__(self,
               concentration,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="Gamma"):
    """Construct Gamma with `concentration` and `rate` parameters.

    The parameters `concentration` and `rate` must be shaped in a way that
    supports broadcasting (e.g. `concentration + rate` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      rate: Floating point tensor, the inverse scale params of the
        distribution(s). Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `concentration` and `rate` are different dtypes.
    """
    parameters = locals()
    with ops.name_scope(name, values=[concentration, rate]) as ns:
      with ops.control_dependencies([
          check_ops.assert_positive(concentration),
          check_ops.assert_positive(rate),
      ] if validate_args else []):
        self._concentration = array_ops.identity(
            concentration, name="concentration")
        self._rate = array_ops.identity(rate, name="rate")
        contrib_tensor_util.assert_same_float_dtype(
            [self._concentration, self._rate])
    super(Gamma, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        is_continuous=True,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._concentration,
                       self._rate],
        name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("concentration", "rate"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.concentration),
        array_ops.shape(self.rate))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.concentration.get_shape(),
        self.rate.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  @distribution_util.AppendDocstring(
      """Note: See `tf.random_gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    return random_ops.random_gamma(
        shape=[n],
        alpha=self.concentration,
        beta=self.rate,
        dtype=self.dtype,
        seed=seed)

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return math_ops.log(self._cdf(x))

  def _cdf(self, x):
    x = self._maybe_assert_valid_sample(x)
    # Note that igamma returns the regularized incomplete gamma function,
    # which is what we want for the CDF.
    return math_ops.igamma(self.concentration, self.rate * x)

  def _log_unnormalized_prob(self, x):
    x = self._maybe_assert_valid_sample(x)
    return (self.concentration - 1.) * math_ops.log(x) - self.rate * x

  def _log_normalization(self):
    return (math_ops.lgamma(self.concentration)
            - self.concentration * math_ops.log(self.rate))

  def _entropy(self):
    return (self.concentration
            - math_ops.log(self.rate)
            + math_ops.lgamma(self.concentration)
            + ((1. - self.concentration) *
               math_ops.digamma(self.concentration)))

  def _mean(self):
    return self.concentration / self.rate

  def _variance(self):
    return self.concentration / math_ops.square(self.rate)

  def _stddev(self):
    return math_ops.sqrt(self.concentration) / self.rate

  @distribution_util.AppendDocstring(
      """The mode of a gamma distribution is `(shape - 1) / rate` when
      `shape > 1`, and `NaN` otherwise. If `self.allow_nan_stats` is `False`,
      an exception will be raised rather than returning `NaN`.""")
  def _mode(self):
    mode = (self.concentration - 1.) / self.rate
    if self.allow_nan_stats:
      nan = array_ops.fill(
          self.batch_shape_tensor(),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      return array_ops.where(self.concentration > 1., mode, nan)
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              array_ops.ones([], self.dtype),
              self.concentration,
              message="mode not defined when any concentration <= 1"),
          ], mode)

  def _maybe_assert_valid_sample(self, x):
    contrib_tensor_util.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(x),
    ], x)


class GammaWithSoftplusConcentrationRate(Gamma):
  """`Gamma` with softplus of `concentration` and `rate`."""

  def __init__(self,
               concentration,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="GammaWithSoftplusConcentrationRate"):
    parameters = locals()
    with ops.name_scope(name, values=[concentration, rate]) as ns:
      super(GammaWithSoftplusConcentrationRate, self).__init__(
          concentration=nn.softplus(concentration,
                                    name="softplus_concentration"),
          rate=nn.softplus(rate, name="softplus_rate"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters


@kullback_leibler.RegisterKL(Gamma, Gamma)
def _kl_gamma_gamma(g0, g1, name=None):
  """Calculate the batched KL divergence KL(g0 || g1) with g0 and g1 Gamma.

  Args:
    g0: instance of a Gamma distribution object.
    g1: instance of a Gamma distribution object.
    name: (optional) Name to use for created operations.
      Default is "kl_gamma_gamma".

  Returns:
    kl_gamma_gamma: `Tensor`. The batchwise KL(g0 || g1).
  """
  with ops.name_scope(name, "kl_gamma_gamma", values=[
      g0.concentration, g0.rate, g1.concentration, g1.rate]):
    # Result from:
    #   http://www.fil.ion.ucl.ac.uk/~wpenny/publications/densities.ps
    # For derivation see:
    #   http://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions   pylint: disable=line-too-long
    return (((g0.concentration - g1.concentration)
             * math_ops.digamma(g0.concentration))
            + math_ops.lgamma(g1.concentration)
            - math_ops.lgamma(g0.concentration)
            + g1.concentration * math_ops.log(g0.rate)
            - g1.concentration * math_ops.log(g1.rate)
            + g0.concentration * (g1.rate / g0.rate - 1.))
