//===- TestMatchers.cpp - Pass to test matchers ---------------------------===//
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

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This is a test pass for verifying matchers.
struct TestMatchers : public ModulePass<TestMatchers> {
  void runOnModule() override;
};
} // end anonymous namespace

// This could be done better but is not worth the variadic template trouble.
template <typename Matcher> unsigned countMatches(FuncOp f, Matcher &matcher) {
  unsigned count = 0;
  f.walk([&count, &matcher](Operation *op) {
    if (matcher.match(op)) {
      // llvm::outs() << "matched " << *op << "\n";
      ++count;
    }
  });
  return count;
}

static void test1(FuncOp f) {
  using mlir::matchers::m_any;
  using mlir::matchers::m_val;

  assert(f.getNumArguments() == 3 && "matcher test funcs must have 3 args");
  auto a = m_val(f.getArgument(0));
  auto b = m_val(f.getArgument(1));
  auto c = m_val(f.getArgument(2));
  (void)a;
  (void)b;
  (void)c;

  llvm::outs() << f.getName();

  auto p0 = m_Op<AddFOp>(); // using 0-arity matcher
  llvm::outs() << "Pattern add(*) matched " << countMatches(f, p0)
               << " times\n";

  auto p1 = m_Op<MulFOp>(); // using 0-arity matcher
  llvm::outs() << "Pattern mul(*) matched " << countMatches(f, p1)
               << " times\n";

  auto p2 = m_Op<AddFOp>(m_Op<AddFOp>(), m_any());
  llvm::outs() << "Pattern add(add(*), *) matched " << countMatches(f, p2)
               << " times\n";

  auto p3 = m_Op<AddFOp>(m_any(), m_Op<AddFOp>());
  llvm::outs() << "Pattern add(*, add(*)) matched " << countMatches(f, p3)
               << " times\n";

  auto p4 = m_Op<MulFOp>(m_Op<AddFOp>(), m_any());
  llvm::outs() << "Pattern mul(add(*), *) matched " << countMatches(f, p4)
               << " times\n";

  auto p5 = m_Op<MulFOp>(m_any(), m_Op<AddFOp>());
  llvm::outs() << "Pattern mul(*, add(*)) matched " << countMatches(f, p5)
               << " times\n";

  auto p6 = m_Op<MulFOp>(m_Op<MulFOp>(), m_any());
  llvm::outs() << "Pattern mul(mul(*), *) matched " << countMatches(f, p6)
               << " times\n";

  auto p7 = m_Op<MulFOp>(m_Op<MulFOp>(), m_Op<MulFOp>());
  llvm::outs() << "Pattern mul(mul(*), mul(*)) matched " << countMatches(f, p7)
               << " times\n";

  auto mul_of_mulmul = m_Op<MulFOp>(m_Op<MulFOp>(), m_Op<MulFOp>());
  auto p8 = m_Op<MulFOp>(mul_of_mulmul, mul_of_mulmul);
  llvm::outs()
      << "Pattern mul(mul(mul(*), mul(*)), mul(mul(*), mul(*))) matched "
      << countMatches(f, p8) << " times\n";

  // clang-format off
  auto mul_of_muladd = m_Op<MulFOp>(m_Op<MulFOp>(), m_Op<AddFOp>());
  auto mul_of_anyadd = m_Op<MulFOp>(m_any(), m_Op<AddFOp>());
  auto p9 = m_Op<MulFOp>(m_Op<MulFOp>(
                     mul_of_muladd, m_Op<MulFOp>()),
                   m_Op<MulFOp>(mul_of_anyadd, mul_of_anyadd));
  // clang-format on
  llvm::outs() << "Pattern mul(mul(mul(mul(*), add(*)), mul(*)), mul(mul(*, "
                  "add(*)), mul(*, add(*)))) matched "
               << countMatches(f, p9) << " times\n";

  auto p10 = m_Op<AddFOp>(a, b);
  llvm::outs() << "Pattern add(a, b) matched " << countMatches(f, p10)
               << " times\n";

  auto p11 = m_Op<AddFOp>(a, c);
  llvm::outs() << "Pattern add(a, c) matched " << countMatches(f, p11)
               << " times\n";

  auto p12 = m_Op<AddFOp>(b, a);
  llvm::outs() << "Pattern add(b, a) matched " << countMatches(f, p12)
               << " times\n";

  auto p13 = m_Op<AddFOp>(c, a);
  llvm::outs() << "Pattern add(c, a) matched " << countMatches(f, p13)
               << " times\n";

  auto p14 = m_Op<MulFOp>(a, m_Op<AddFOp>(c, b));
  llvm::outs() << "Pattern mul(a, add(c, b)) matched " << countMatches(f, p14)
               << " times\n";

  auto p15 = m_Op<MulFOp>(a, m_Op<AddFOp>(b, c));
  llvm::outs() << "Pattern mul(a, add(b, c)) matched " << countMatches(f, p15)
               << " times\n";

  auto mul_of_aany = m_Op<MulFOp>(a, m_any());
  auto p16 = m_Op<MulFOp>(mul_of_aany, m_Op<AddFOp>(a, c));
  llvm::outs() << "Pattern mul(mul(a, *), add(a, c)) matched "
               << countMatches(f, p16) << " times\n";

  auto p17 = m_Op<MulFOp>(mul_of_aany, m_Op<AddFOp>(c, b));
  llvm::outs() << "Pattern mul(mul(a, *), add(c, b)) matched "
               << countMatches(f, p17) << " times\n";
}

void TestMatchers::runOnModule() {
  auto m = getModule();
  for (auto f : m.getOps<FuncOp>()) {
    if (f.getName() == "test1")
      test1(f);
  }
}

static PassRegistration<TestMatchers> pass("test-matchers",
                                           "Test C++ pattern matchers.");
