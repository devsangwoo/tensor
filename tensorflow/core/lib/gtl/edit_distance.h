<<<<<<< HEAD
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LIB_GTL_EDIT_DISTANCE_H_
#define TENSORFLOW_LIB_GTL_EDIT_DISTANCE_H_

#include <numeric>

=======
#ifndef TENSORFLOW_LIB_GTL_EDIT_DISTANCE_H_
#define TENSORFLOW_LIB_GTL_EDIT_DISTANCE_H_

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {
namespace gtl {

// Calculate the Levenshtein Edit Distance between two contiguous
// sequences, s and t, of type T.
//
// The Levenshtein distance is a symmetric distance defined as the
// smallest number of insertions, deletions, and substitutions
// required to convert sequence s to t (and vice versa).
// Note, this distance does not consider transpositions.
//
// For more details and a reference implementation, see:
//   https://en.wikipedia.org/wiki/Levenshtein_distance
//
// This implementation has time complexity O(|s|*|t|)
// and space complexity O(min(|s|, |t|)), where
//   |x| := x.size()
//
// A simple call to LevenshteinDistance looks like:
//
//  int64 dist = LevenshteinDistance("hi", "bye", std::equal_to<char>());
//
template <typename T, typename Cmp>
inline int64 LevenshteinDistance(const gtl::ArraySlice<T>& s,
                                 const gtl::ArraySlice<T>& t, const Cmp& cmp) {
  const int64 s_size = s.size();
  const int64 t_size = t.size();

<<<<<<< HEAD
  if (t_size > s_size) return LevenshteinDistance(t, s, cmp);

  const T* s_data = s.data();
  const T* t_data = t.data();

  if (t_size == 0) return s_size;
  if (s == t) return 0;

  // Create work vector
  gtl::InlinedVector<int64, 32> scratch_holder(t_size);

  int64* scratch = scratch_holder.data();

  // Special case for i = 0: Distance between empty string and string
  // of length j is just j.
  for (size_t j = 1; j < t_size; ++j) scratch[j - 1] = j;

  for (size_t i = 1; i <= s_size; ++i) {
    // Invariant: scratch[j - 1] equals cost(i - 1, j).
    int substitution_base_cost = i - 1;
    int insertion_cost = i + 1;
    for (size_t j = 1; j <= t_size; ++j) {
      // Invariants:
      //  scratch[k - 1] = cost(i, k)  for 0 < k < j.
      //  scratch[k - 1] = cost(i - 1, k)  for j <= k <= t_size.
      //  substitution_base_cost = cost(i - 1, j - 1)
      //  insertion_cost = cost(i, j - 1)
      const int replacement_cost = cmp(s_data[i - 1], t_data[j - 1]) ? 0 : 1;
      const int substitution_cost = substitution_base_cost + replacement_cost;
      const int deletion_cost = scratch[j - 1] + 1;

      // Select the cheapest edit.
      const int cheapest =  // = cost(i, j)
          std::min(deletion_cost, std::min(insertion_cost, substitution_cost));

      // Restore invariant for the next iteration of the loop.
      substitution_base_cost = scratch[j - 1];  // = cost(i - 1, j)
      scratch[j - 1] = cheapest;                // = cost(i, j)
      insertion_cost = cheapest + 1;            // = cost(i, j) + 1
    }
  }
  return scratch[t_size - 1];
=======
  if (s_size == 0) return t_size;
  if (t_size == 0) return s_size;
  if (s == t) return 0;
  if (t_size > s_size) return LevenshteinDistance(t, s, cmp);

  // Create work vectors
  gtl::InlinedVector<int64, 32> scratch0(t_size + 1);
  gtl::InlinedVector<int64, 32> scratch1(t_size + 1);

  int64* previous = scratch0.data();
  int64* current = scratch1.data();

  // Initialize previous row of distances
  std::iota(scratch0.begin(), scratch0.end(), 0);

  for (int64 i = 0; i < s_size; ++i) {
    // Swap current and previous rows for next iteration
    std::swap(previous, current);

    // Calculate current row distances from previous row
    current[0] = i + 1;

    // Fill in the rest of the row
    for (int64 j = 0; j < t_size; ++j) {
      const int64 cost = cmp(s[i], t[j]) ? 0 : 1;
      current[j + 1] =
          std::min(current[j] + 1,                 // deletion cost
                   std::min(previous[j + 1] + 1,   // insertion cost
                            previous[j] + cost));  // substitution cost
    }
  }

  return current[t_size];
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
}

template <typename Container1, typename Container2, typename Cmp>
inline int64 LevenshteinDistance(const Container1& s, const Container2& t,
                                 const Cmp& cmp) {
  return LevenshteinDistance(
      gtl::ArraySlice<typename Container1::value_type>(s.data(), s.size()),
      gtl::ArraySlice<typename Container1::value_type>(t.data(), t.size()),
      cmp);
}

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_GTL_EDIT_DISTANCE_H_
