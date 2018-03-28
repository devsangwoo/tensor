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
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
#
# Do not use pylint on generated code.
# pylint: disable=missing-docstring,g-short-docstring-punctuation,g-no-space-after-docstring-summary,invalid-name,line-too-long,unused-argument,g-doc-args
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import grpc

from third_party.tensorflow.contrib.tpu.profiler import tpu_profiler_analysis_pb2 as third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2


class TPUProfileAnalysisStub(object):
  """//////////////////////////////////////////////////////////////////////////////

  TPUProfileAnalysis service provide entry point for profiling TPU and for
  serving profiled data to Tensorboard through GRPC
  //////////////////////////////////////////////////////////////////////////////
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.NewSession = channel.unary_unary(
        '/tensorflow.TPUProfileAnalysis/NewSession',
        request_serializer=
        third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
        NewProfileSessionRequest.SerializeToString,
        response_deserializer=
        third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
        NewProfileSessionResponse.FromString,
    )
    self.EnumSessions = channel.unary_unary(
        '/tensorflow.TPUProfileAnalysis/EnumSessions',
        request_serializer=
        third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
        EnumProfileSessionsAndToolsRequest.SerializeToString,
        response_deserializer=
        third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
        EnumProfileSessionsAndToolsResponse.FromString,
    )
    self.GetSessionToolData = channel.unary_unary(
        '/tensorflow.TPUProfileAnalysis/GetSessionToolData',
        request_serializer=
        third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
        ProfileSessionDataRequest.SerializeToString,
        response_deserializer=
        third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
        ProfileSessionDataResponse.FromString,
    )


class TPUProfileAnalysisServicer(object):
  """//////////////////////////////////////////////////////////////////////////////

  TPUProfileAnalysis service provide entry point for profiling TPU and for
  serving profiled data to Tensorboard through GRPC
  //////////////////////////////////////////////////////////////////////////////
  """

  def NewSession(self, request, context):
    """Starts a profiling session, blocks until it completes.
    TPUProfileAnalysis service delegate this to TPUProfiler service.
    Populate the profiled data in repository, then return status to caller.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def EnumSessions(self, request, context):
    """Enumerate existing sessions and return available profile tools.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetSessionToolData(self, request, context):
    """Retrieve specific tool's data for specific session.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_TPUProfileAnalysisServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'NewSession':
          grpc.unary_unary_rpc_method_handler(
              servicer.NewSession,
              request_deserializer=
              third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
              NewProfileSessionRequest.FromString,
              response_serializer=
              third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
              NewProfileSessionResponse.SerializeToString,
          ),
      'EnumSessions':
          grpc.unary_unary_rpc_method_handler(
              servicer.EnumSessions,
              request_deserializer=
              third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
              EnumProfileSessionsAndToolsRequest.FromString,
              response_serializer=
              third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
              EnumProfileSessionsAndToolsResponse.SerializeToString,
          ),
      'GetSessionToolData':
          grpc.unary_unary_rpc_method_handler(
              servicer.GetSessionToolData,
              request_deserializer=
              third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
              ProfileSessionDataRequest.FromString,
              response_serializer=
              third__party_dot_tensorflow_dot_contrib_dot_tpu_dot_profiler_dot_tpu__profiler__analysis__pb2.
              ProfileSessionDataResponse.SerializeToString,
          ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tensorflow.TPUProfileAnalysis', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
