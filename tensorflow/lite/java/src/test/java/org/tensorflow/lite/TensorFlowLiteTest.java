/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.lite.TensorFlowLite}. */
@RunWith(JUnit4.class)
public final class TensorFlowLiteTest {

  @Test
  @SuppressWarnings("deprecation")
  public void testVersion() {
    assertThat(TensorFlowLite.version()).isEqualTo("3");
  }

  @Test
  public void testSchemaVersion() {
    assertThat(TensorFlowLite.schemaVersion()).isEqualTo("3");
  }

  @Test
  public void testRuntimeVersion() {
    assertThat(TensorFlowLite.runtimeVersion()).startsWith("1.");
  }
}
