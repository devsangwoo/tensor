//===- Module.h - MLIR Module Class -----------------------------*- C++ -*-===//
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
//
// Module is the top-level container for code in an MLIR program.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MODULE_H
#define MLIR_IR_MODULE_H

#include "mlir/IR/Function.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ilist.h"
#include <vector>

namespace mlir {

class AffineMap;

class Module {
public:
  explicit Module(MLIRContext *context);

  MLIRContext *getContext() const { return context; }

  /// This is the list of functions in the module.
  typedef llvm::iplist<Function> FunctionListType;
  FunctionListType &getFunctions() { return functions; }
  const FunctionListType &getFunctions() const { return functions; }

  // Iteration over the functions in the module.
  using iterator = FunctionListType::iterator;
  using const_iterator = FunctionListType::const_iterator;
  using reverse_iterator = FunctionListType::reverse_iterator;
  using const_reverse_iterator = FunctionListType::const_reverse_iterator;

  iterator begin() { return functions.begin(); }
  iterator end() { return functions.end(); }
  const_iterator begin() const { return functions.begin(); }
  const_iterator end() const { return functions.end(); }
  reverse_iterator rbegin() { return functions.rbegin(); }
  reverse_iterator rend() { return functions.rend(); }
  const_reverse_iterator rbegin() const { return functions.rbegin(); }
  const_reverse_iterator rend() const { return functions.rend(); }

  // Interfaces for working with the symbol table.

  /// Look up a function with the specified name, returning null if no such
  /// name exists.  Function names never include the @ on them.
  Function *getNamedFunction(StringRef name);
  const Function *getNamedFunction(StringRef name) const {
    return const_cast<Module *>(this)->getNamedFunction(name);
  }

  /// Look up a function with the specified name, returning null if no such
  /// name exists.  Function names never include the @ on them.
  Function *getNamedFunction(Identifier name);

  const Function *getNamedFunction(Identifier name) const {
    return const_cast<Module *>(this)->getNamedFunction(name);
  }

  /// Perform (potentially expensive) checks of invariants, used to detect
  /// compiler bugs.  On error, this fills in the string and return true,
  /// or aborts if the string was not provided.
  bool verify(std::string *errorResult = nullptr) const;

  void print(raw_ostream &os) const;
  void dump() const;

private:
  friend struct llvm::ilist_traits<Function>;

  /// getSublistAccess() - Returns pointer to member of function list
  static FunctionListType Module::*getSublistAccess(Function *) {
    return &Module::functions;
  }

  MLIRContext *context;

  /// This is a mapping from a name to the function with that name.
  llvm::DenseMap<Identifier, Function *> symbolTable;

  /// This is used when name conflicts are detected.
  unsigned uniquingCounter = 0;

  /// This is the actual list of functions the module contains.
  FunctionListType functions;
};
} // end namespace mlir

#endif  // MLIR_IR_FUNCTION_H
