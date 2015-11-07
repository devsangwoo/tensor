#ifndef TENSORFLOW_KERNELS_READER_BASE_H_
#define TENSORFLOW_KERNELS_READER_BASE_H_

#include <memory>
#include <string>
#include <vector>
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/reader_interface.h"
#include "tensorflow/core/kernels/reader_base.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

// Default implementation of ReaderInterface.
class ReaderBase : public ReaderInterface {
 public:
  // name: For use in error messages, should mention both the name of
  // the op and the node.
  explicit ReaderBase(const string& name);

  // Note that methods with names ending in "Locked" are called while
  // the ReaderBase's mutex is held.

  // Implement this function in descendants -----------------------------------

  // Produce the next key/value pair from the current work item.
  // This is called "Locked" since it is executed under a mutex
  // that serializes all Reader calls.
  // Usage:
  //  a) If a record was successfully produced, set *produced = true,
  //  and fill in *key and *value.
  //  b) If no more records will be produced for this work item, set
  //  *at_end = true.
  //  c) If a record was produced, but no more will be produced, you
  //     may either do both (a) and (b), or do (a) in this call and do (b) in
  //     the next call to ReadLocked().
  //  d) If there was an error producing (e.g. an error reading the file,
  //     data corruption), return a non-OK() status.  ReadLocked may be
  //     called again if the user reruns this part of the graph.
  virtual Status ReadLocked(string* key, string* value, bool* produced,
                            bool* at_end) = 0;

  // Descendants may optionally implement these -------------------------------

  // Called when work starts / finishes.
  virtual Status OnWorkStartedLocked() { return Status::OK(); }
  virtual Status OnWorkFinishedLocked() { return Status::OK(); }

  // Called to reset the Reader to a newly constructed state.
  virtual Status ResetLocked();

  // Default implementation generates an Unimplemented error.
  // See the protected helper methods below.
  virtual Status SerializeStateLocked(string* state);
  virtual Status RestoreStateLocked(const string& state);

  // Accessors ----------------------------------------------------------------

  // Always true during a call to ReadLocked().
  bool work_in_progress() const { return work_finished_ < work_started_; }

  // Returns the name of the current work item (valid if
  // work_in_progress() returns true).  May change between calls to
  // ReadLocked().
  const string& current_work() const { return work_; }

  // What was passed to the constructor.
  const string& name() const { return name_; }

 protected:
  // For descendants wishing to implement serialize & restore state.

  // Writes ReaderBase state to *state.
  void SaveBaseState(ReaderBaseState* state) const;

  // Restores ReaderBase state from state. Assumes state was filled
  // using SaveBaseState() above.
  Status RestoreBaseState(const ReaderBaseState& state);

 private:
  // Implementations of ReaderInterface methods.  These ensure thread-safety
  // and call the methods above to do the work.
  void Read(QueueInterface* queue, string* key, string* value,
            OpKernelContext* context) override;
  Status Reset() override;
  int64 NumRecordsProduced() override;
  int64 NumWorkUnitsCompleted() override;
  Status SerializeState(string* state) override;
  Status RestoreState(const string& state) override;

  // For implementing Read().  Dequeues the next work item from
  // *queue, and if successful updates work_, work_started_
  // (establishing work_in_progress() == true) and calls
  // OnWorkStartedLocked().  May block.
  void GetNextWorkLocked(QueueInterface* queue, OpKernelContext* context);

  mutable mutex mu_;
  const string name_;
  int64 work_started_ = 0;
  int64 work_finished_ = 0;
  int64 num_records_produced_ = 0;
  string work_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_READER_BASE_H_
