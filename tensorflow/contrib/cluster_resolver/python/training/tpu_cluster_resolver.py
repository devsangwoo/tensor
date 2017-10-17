# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of Cluster Resolvers for Cloud TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.cluster_resolver.python.training.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec

_GOOGLE_API_CLIENT_INSTALLED = True
try:
  from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
  from oauth2client.client import GoogleCredentials  # pylint: disable=g-import-not-at-top
except ImportError:
  _GOOGLE_API_CLIENT_INSTALLED = False


class TPUClusterResolver(ClusterResolver):
  """Cluster Resolver for Google Cloud TPUs.

  This is an implementation of cluster resolvers for the Google Cloud TPU
  service. As Cloud TPUs are in alpha, you will need to specify a API definition
  file for this to consume, in addition to a list of Cloud TPUs in your Google
  Cloud Platform project.
  """

  def __init__(self,
               project,
               zone,
               tpu_names,
               job_name='tpu_worker',
               credentials='default',
               service=None):
    """Creates a new TPUClusterResolver object.

    The ClusterResolver will then use the parameters to query the Cloud TPU APIs
    for the IP addresses and ports of each Cloud TPU listed.

    Args:
      project: Name of the GCP project containing Cloud TPUs
      zone: Zone where the TPUs are located
      tpu_names: A list of names of the target Cloud TPUs.
      job_name: Name of the TensorFlow job the TPUs belong to.
      credentials: GCE Credentials. If None, then we use default credentials
        from the oauth2client
      service: The GCE API object returned by the googleapiclient.discovery
        function. If you specify a custom service object, then the credentials
        parameter will be ignored.

    Raises:
      ImportError: If the googleapiclient is not installed.
    """

    self._project = project
    self._zone = zone
    self._tpu_names = tpu_names
    self._job_name = job_name
    self._credentials = credentials

    if credentials == 'default':
      if _GOOGLE_API_CLIENT_INSTALLED:
        self._credentials = GoogleCredentials.get_application_default()

    if service is None:
      if not _GOOGLE_API_CLIENT_INSTALLED:
        raise ImportError('googleapiclient must be installed before using the '
                          'TPU cluster resolver')

      # TODO(b/67375680): Remove custom URL once TPU APIs are finalized
      self._service = discovery.build(
          'tpu',
          'v1',
          credentials=self._credentials,
          discoveryServiceUrl='https://storage.googleapis.com'
                              '/tpu-api-definition/v1alpha1.json')
    else:
      self._service = service

  def cluster_spec(self):
    """Returns a ClusterSpec object based on the latest TPU information.

    We retrieve the information from the GCE APIs every time this method is
    called.

    Returns:
      A ClusterSpec containing host information returned from Cloud TPUs.
    """
    worker_list = []

    for tpu_name in self._tpu_names:
      full_name = 'projects/%s/locations/%s/nodes/%s' % (
          self._project, self._zone, tpu_name)
      request = self._service.projects().locations().nodes().get(name=full_name)
      response = request.execute()

      instance_url = '%s:%s' % (response['ipAddress'], response['port'])
      worker_list.append(instance_url)

    return ClusterSpec({self._job_name: worker_list})
