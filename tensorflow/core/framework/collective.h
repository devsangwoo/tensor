/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_FRAMEWORK_COLLECTIVE_EXECUTOR_H_
#define TENSORFLOW_FRAMEWORK_COLLECTIVE_EXECUTOR_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class BufRendezvous;
class CancellationManager;
class CompleteGroupRequest;
class CompleteGroupResponse;
class CompleteInstanceRequest;
class CompleteInstanceResponse;
class DeviceLocality;
class GetStepSequenceRequest;
class GetStepSequenceResponse;
class Op;
class Tensor;

// Types of supported collective operations.
enum CollectiveType {
  REDUCTION_COLLECTIVE = 0,
  BROADCAST_COLLECTIVE,
  UNDEFINED_COLLECTIVE,
};

// Data common to all members of a device group.
// All members share the same device set but its order is
// particular to an instance so it is stored there.
struct CollGroupParams {
  int32 group_key;
  int32 group_size;
  DeviceType device_type;
  int32 num_tasks;  // number of distinct tasks in group
  string ToString() const;
  CollGroupParams() : device_type(DEVICE_CPU) {}
};

// The best implementation of a collective op depends on many factors
// including the number of devices involved, the topology of
// interconnects between them and the sizes of inputs.  This structure
// is used in generating and representing data movement choreography
// for each specific algorithm, hence it does not have a single, fixed
// interpretation.  On first execution the runtime will update this
// structure with decisions that will guide all subsequent executions.
struct CollImplDetails {
  std::vector<std::vector<int>> subdiv_permutations;
  std::vector<int> subdiv_offsets;
  // broadcast only: rank of source in each subdiv
  std::vector<int> subdiv_source_rank;
};

// Data common to all members of a collective instance.
struct CollInstanceParams {
  int32 instance_key;  // Identifies all participating graph nodes.
  CollectiveType type;
  DataType data_type;
  TensorShape shape;
  // Fully qualified name of device for each member, in default rank order.
  std::vector<string> device_names;
  // Task name prefix of corresponding device name.
  std::vector<string> task_names;
  // True if every task has the same number of devices.
  bool same_num_devices_per_task;
  CollImplDetails impl_details;
  string ToString() const;
  CollInstanceParams& operator=(const struct CollInstanceParams& other);
};

// Data common to all instance members in the same task.
struct CollTaskParams {
  // True for devices that are local to the process, i.e. no RPC needed.
  std::vector<bool> is_local;
  string ToString() const;
};

// Unique to a single CollectiveOp node.
struct CollectiveParams {
  CollGroupParams group;
  CollInstanceParams instance;
  CollTaskParams task;

  string name;       // node name used only for log or error messages
  int default_rank;  // index of this op within device_names
  bool is_source;    // broadcast only
  // Rank of this device in each subdivision permutation.
  std::vector<int> subdiv_rank;
  std::unique_ptr<OpKernel> merge_op;  // reduction only
  std::unique_ptr<OpKernel> final_op;  // reduction only
  string ToString() const;
};

class CollectiveExecutor;

// Interface that provides resolution of device localities.
class DeviceResolverInterface {
 public:
  virtual ~DeviceResolverInterface() {}

  // Collects DeviceLocality protobufs from all of the devices identified
  // in 'col_params'.
  virtual void GetDeviceLocalitiesAsync(const CollInstanceParams& inst_params,
                                        std::vector<DeviceLocality>* localities,
                                        const StatusCallback& done) = 0;

  // Populate *locality with the DeviceLocality of the specified
  // device.
  virtual void GetLocalityAsync(const string& device, const string& task,
                                DeviceLocality* locality,
                                const StatusCallback& done) = 0;

  // Clear the cache of device data belonging
  // to the specified task.
  virtual void ClearTask(const string& task) = 0;
};

// Interface that provides resolution of shared CollectiveParams fields.
class ParamResolverInterface {
 public:
  virtual ~ParamResolverInterface() {}

  // Called by each collective op at first execution in order to fill out
  // the CollectiveParams structure with data gathered from the full
  // (maybe distributed) collection of peer nodes.
  virtual void CompleteParamsAsync(const string& device, CollectiveParams* cp,
                                   CancellationManager* cancel_mgr,
                                   const StatusCallback& done) = 0;

  // Used within a distributed implementation to discover/verify
  // data shared across a device group.
  virtual void CompleteGroupAsync(const CompleteGroupRequest* request,
                                  CompleteGroupResponse* response,
                                  CancellationManager* cancel_mgr,
                                  const StatusCallback& done) = 0;

  // Used within a distributed implementation to discover/verify data
  // shared across an instance group.
  virtual void CompleteInstanceAsync(const CompleteInstanceRequest* request,
                                     CompleteInstanceResponse* response,
                                     CancellationManager* cancel_mgr,
                                     const StatusCallback& done) = 0;
};

// Graphs which utilize Collective Ops in a common instance must
// execute with identical step_ids even if they are disjoint graphs
// run by otherwise independent tasks.  This interface supplies
// coordinated step_ids to use in such cases.
class StepSequenceInterface {
 public:
  virtual ~StepSequenceInterface() {}

  // Used with a distributed implementation to coordinate step_id
  // sequences across tasks.
  virtual void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                                    GetStepSequenceResponse* response,
                                    const StatusCallback& done) = 0;

  // Refresh the local per-graph_key step_id sequence from collective
  // group leader, if applicable.
  virtual void RefreshStepIdSequenceAsync(int64 graph_key,
                                          const StatusCallback& done) = 0;

  // Returns the the step_id that should be used for initiating a new execution
  // on the specified graph. May return the same step_id multiple times if
  // RetireStepId or RefreshStepIdReservation is not called.
  virtual int64 NextStepId(int64 graph_key) = 0;

  // Reports that execution of the given step has completed successfully.
  // Should be called immediately after a step completes with OK status,
  // prior to calling NextStepId().  If the step fails, don't call.
  virtual void RetireStepId(int64 graph_key, int64 step_id) = 0;
};

// Interface that provides access to per-step CollectiveExecutor
// instances and various distributed resolution capabilities.
class CollectiveExecutorMgrInterface : public StepSequenceInterface {
 public:
  virtual ~CollectiveExecutorMgrInterface() {}

  // Returns the step-specific CollectiveExecutor, creating if one does not
  // already exist.  The caller assumes ownership of one Ref on the object.
  virtual CollectiveExecutor* FindOrCreate(int64 step_id) = 0;

  // If there is a CollectiveExecutor for step_id, remove it from the
  // table.
  virtual void Cleanup(int64 step_id) = 0;

  virtual ParamResolverInterface* GetParamResolver() const = 0;

  virtual DeviceResolverInterface* GetDeviceResolver() const = 0;
};

// Interface that a Collective Op implementation uses to exchange data
// with peers.  Note that data exchange is currently limited to types
// for which DMAHelper::CanUseDMA() returns true, i.e.  dense numeric
// types.
class PeerAccessInterface {
 public:
  virtual ~PeerAccessInterface() {}

  virtual void RecvFromPeer(const string& peer_device, const string& peer_task,
                            bool peer_is_local, const string& key,
                            Device* to_device, DeviceContext* to_device_ctx,
                            const AllocatorAttributes& to_alloc_attr,
                            Tensor* to_tensor,
                            const DeviceLocality& client_locality,
                            const StatusCallback& done) = 0;

  virtual void PostToPeer(const string& peer_device, const string& peer_task,
                          const string& key, Device* from_device,
                          DeviceContext* from_device_ctx,
                          const AllocatorAttributes& from_alloc_attr,
                          const Tensor* from_tensor,
                          const DeviceLocality& client_locality,
                          const StatusCallback& done) = 0;
};

class PerStepCollectiveRemoteAccess;

// A step-specific object that can execute a collective operation completely
// described by a CollectiveParams object.
class CollectiveExecutor : public PeerAccessInterface, public core::RefCounted {
 public:
  virtual void StartAbort(const Status& s) {}

  virtual void ExecuteAsync(OpKernelContext* ctx,
                            const CollectiveParams& col_params,
                            const string& exec_key, StatusCallback done) {
    done(errors::Internal(
        "A collective Op has been called in a context in which "
        "a CollectiveExecutor has not been provided."));
  }

  virtual void CompleteParamsAsync(const string& device, CollectiveParams* cp,
                                   CancellationManager* cancel_mgr,
                                   StatusCallback done) {
    cem_->GetParamResolver()->CompleteParamsAsync(device, cp, cancel_mgr, done);
  }

  virtual PerStepCollectiveRemoteAccess* remote_access() { return nullptr; }

  // Used to designate an invalid group or instance key.
  static int64 kInvalidId;

  // Lexically scoped handle for Ref.
  class Handle {
   public:
    explicit Handle(CollectiveExecutor* ce, bool inherit_ref) : ce_(ce) {
      if (!inherit_ref) ce->Ref();
    }
    ~Handle() { ce_->Unref(); }
    CollectiveExecutor* get() const { return ce_; }

   private:
    CollectiveExecutor* ce_;
  };

 protected:
  explicit CollectiveExecutor(CollectiveExecutorMgrInterface* cem)
      : cem_(cem) {}

  // For use only by derived classes
  static OpKernelContext::Params* CtxParams(OpKernelContext* ctx);
  CollectiveExecutorMgrInterface* cem_;

  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveExecutor);
};

// Interface of a helper object that provides a CollectiveExecutor with
// all of the remote access it needs.
class CollectiveRemoteAccess : public PeerAccessInterface,
                               public DeviceResolverInterface {
 public:
  virtual ~CollectiveRemoteAccess() {}

  virtual BufRendezvous* buf_rendezvous() = 0;
};

// A per-step version of CollectiveRemoteAccess that cleans up outstanding
// communications in case step execution is abandoned.
class PerStepCollectiveRemoteAccess : public CollectiveRemoteAccess {
 public:
  virtual ~PerStepCollectiveRemoteAccess() {}
  virtual void StartAbort(const Status& s) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_COLLECTIVE_EXECUTOR_H_
