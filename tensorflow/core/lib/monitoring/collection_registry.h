/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
#define THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_

#include <map>
#include <memory>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace monitoring {

namespace internal {
class Collector;
}  // namespace internal

// Metric implementations would get an instance of this class using the
// MetricCollectorGetter in the collection-function lambda, so that their values
// can be collected.
//
// Read the documentation on CollectionRegistry::Register() for more details.
//
// For example:
//   auto metric_collector = metric_collector_getter->Get(&metric_def);
//   metric_collector.CollectValue(some_labels, some_value);
//   metric_collector.CollectValue(others_labels, other_value);
//
// This class is NOT thread-safe.
template <MetricKind metric_kind, typename Value, int NumLabels>
class MetricCollector {
 public:
  ~MetricCollector() = default;

  // Collects the value with these labels.
  void CollectValue(const std::array<string, NumLabels>& labels,
                    const Value& value);

 private:
  friend class internal::Collector;

  MetricCollector(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def,
      PointSet* const point_set)
      : metric_def_(metric_def), point_set_(point_set) {
    point_set_->metric_name = metric_def->name().ToString();
  }

  const MetricDef<metric_kind, Value, NumLabels>* const metric_def_;
  PointSet* const point_set_;

  // This is made copyable because we can't hand out references of this class
  // from MetricCollectorGetter because this class is templatized, and we need
  // MetricCollectorGetter not to be templatized and hence MetricCollectorGetter
  // can't own an instance of this class.
};

// Returns a MetricCollector with the same template parameters as the
// metric-definition, so that the values of a metric can be collected.
//
// The collection-function defined by a metric takes this as a parameter.
//
// Read the documentation on CollectionRegistry::Register() for more details.
class MetricCollectorGetter {
 public:
  // Returns the MetricCollector with the same template parameters as the
  // metric_def.
  template <MetricKind metric_kind, typename Value, int NumLabels>
  MetricCollector<metric_kind, Value, NumLabels> Get(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def);

 private:
  friend class internal::Collector;

  MetricCollectorGetter(internal::Collector* const collector,
                        const AbstractMetricDef* const allowed_metric_def)
      : collector_(collector), allowed_metric_def_(allowed_metric_def) {}

  internal::Collector* const collector_;
  const AbstractMetricDef* const allowed_metric_def_;
};

// A collection registry for metrics.
//
// Metrics are registered here so that their state can be collected later and
// exported.
//
// This class is thread-safe.
class CollectionRegistry {
 public:
  ~CollectionRegistry() = default;

  // Returns the default registry for the process.
  //
  // This registry belongs to this library and should never be deleted.
  static CollectionRegistry* Default();

  using CollectionFunction = std::function<void(MetricCollectorGetter getter)>;

  // Registers the metric and the collection-function which can be used to
  // collect its values. Returns a Registration object, which when upon
  // destruction would cause the metric to be unregistered from this registry.
  //
  // IMPORTANT: Delete the handle before the metric-def is deleted.
  //
  // Example usage;
  // CollectionRegistry::Default()->Register(
  //   &metric_def,
  //   [&](MetricCollectorGetter getter) {
  //     auto metric_collector = getter.Get(&metric_def);
  //     for (const auto& cell : cells) {
  //       metric_collector.CollectValue(cell.labels(), cell.value());
  //     }
  //   });
  class RegistrationHandle;
  std::unique_ptr<RegistrationHandle> Register(
      const AbstractMetricDef* metric_def,
      const CollectionFunction& collection_function)
      LOCKS_EXCLUDED(mu_) TF_MUST_USE_RESULT;

  // Goes through all the registered metrics, collects their current values and
  // returns them as the CollectedMetrics proto.
  //
  // TODO(vinuraja): Add support for options whereby we can skip filling
  // MetricDescriptors, because we just need to collect it once during the
  // process lifetime.
  std::unique_ptr<CollectedMetrics> CollectMetrics();

 private:
  CollectionRegistry() = default;

  // Unregisters the metric from this registry. This is private because the
  // public interface provides a Registration handle which automatically calls
  // this upon destruction.
  void Unregister(const AbstractMetricDef* metric_def) LOCKS_EXCLUDED(mu_);

  mutable mutex mu_;

  struct RegistrationInfo {
    const AbstractMetricDef* const metric_def;
    CollectionFunction collection_function;
  };
  std::map<StringPiece, RegistrationInfo> registry_ GUARDED_BY(mu_);
};

////
// Implementation details follow. API readers may skip.
////

class CollectionRegistry::RegistrationHandle {
 public:
  RegistrationHandle(CollectionRegistry* const export_registry,
                     const AbstractMetricDef* const metric_def)
      : export_registry_(export_registry), metric_def_(metric_def) {}

  ~RegistrationHandle() { export_registry_->Unregister(metric_def_); }

 private:
  CollectionRegistry* const export_registry_;
  const AbstractMetricDef* const metric_def_;
};

namespace internal {

template <typename Value>
void CollectValue(const Value& value, Point* point);

template <>
inline void CollectValue(const int64& value, Point* const point) {
  point->value_type = ValueType::kInt64;
  point->int64_value = value;
}

// Used by the CollectionRegistry class to collect all the values of all the
// metrics in the registry. This is an implementation detail of the
// CollectionRegistry class, please do not depend on this.
//
// This cannot be a private nested class because we need to forward declare this
// so that the MetricCollector and MetricCollectorGetter classes can be friends
// with it.
//
// This class is thread-safe.
class Collector {
 public:
  Collector() : collected_metrics_(new CollectedMetrics()) {}

  template <MetricKind metric_kind, typename Value, int NumLabels>
  MetricCollector<metric_kind, Value, NumLabels> GetMetricCollector(
      const MetricDef<metric_kind, Value, NumLabels>* const metric_def)
      LOCKS_EXCLUDED(mu_) {
    CollectMetricDescriptor(metric_def);

    auto* const point_set = [&]() {
      mutex_lock l(mu_);
      return collected_metrics_->point_set_map
          .insert(std::make_pair(metric_def->name().ToString(),
                                 std::unique_ptr<PointSet>(new PointSet())))
          .first->second.get();
    }();
    return MetricCollector<metric_kind, Value, NumLabels>(metric_def,
                                                          point_set);
  }

  void CollectMetric(
      const AbstractMetricDef* const metric_def,
      const CollectionRegistry::CollectionFunction& collection_function);

  std::unique_ptr<CollectedMetrics> ConsumeCollectedMetrics()
      LOCKS_EXCLUDED(mu_);

  void CollectMetricDescriptor(const AbstractMetricDef* const metric_def)
      LOCKS_EXCLUDED(mu_);

 private:
  mutable mutex mu_;
  std::unique_ptr<CollectedMetrics> collected_metrics_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(Collector);
};

}  // namespace internal

template <MetricKind metric_kind, typename Value, int NumLabels>
void MetricCollector<metric_kind, Value, NumLabels>::CollectValue(
    const std::array<string, NumLabels>& labels, const Value& value) {
  point_set_->points.emplace_back(new Point());
  auto* const point = point_set_->points.back().get();
  const std::vector<StringPiece> label_descriptions =
      metric_def_->label_descriptions();
  point->labels.reserve(NumLabels);
  for (int i = 0; i < NumLabels; ++i) {
    point->labels.push_back({});
    auto* const label = &point->labels.back();
    label->name = label_descriptions[i].ToString();
    label->value = labels[i];
  }
  internal::CollectValue(value, point);
  // TODO(vinuraja): Implement timestamp collection too.
}

template <MetricKind metric_kind, typename Value, int NumLabels>
MetricCollector<metric_kind, Value, NumLabels> MetricCollectorGetter::Get(
    const MetricDef<metric_kind, Value, NumLabels>* const metric_def) {
  if (allowed_metric_def_ != metric_def) {
    LOG(FATAL) << "Expected collection for: " << allowed_metric_def_->name()
               << " but instead got: " << metric_def->name();
  }

  return collector_->GetMetricCollector(metric_def);
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
