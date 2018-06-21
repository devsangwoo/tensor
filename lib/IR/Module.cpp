//===- Module.cpp - MLIR Module Class -------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

Module::Module() {
}


void Module::print(raw_ostream &os) {
  for (auto *fn : functionList)
    fn->print(os);
}

void Module::dump() {
  print(llvm::errs());
}

