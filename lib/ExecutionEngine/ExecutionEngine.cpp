//===- ExecutionEngine.cpp - MLIR Execution engine and utils --------------===//
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
// This file implements the execution engine for MLIR modules based on LLVM Orc
// JIT engine.
//
//===----------------------------------------------------------------------===//
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/LLVMIR/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"

using namespace mlir;
using llvm::Error;
using llvm::Expected;

namespace {
// Memory manager for the JIT's objectLayer.  Its main goal is to fallback to
// resolving functions in the current process if they cannot be resolved in the
// JIT-compiled modules.
class MemoryManager : public llvm::SectionMemoryManager {
public:
  MemoryManager(llvm::orc::ExecutionSession &execSession)
      : session(execSession) {}

  // Resolve the named symbol.  First, try looking it up in the main library of
  // the execution session.  If there is no such symbol, try looking it up in
  // the current process (for example, if it is a standard library function).
  // Return `nullptr` if lookup fails.
  llvm::JITSymbol findSymbol(const std::string &name) override {
    auto mainLibSymbol = session.lookup({&session.getMainJITDylib()}, name);
    if (mainLibSymbol)
      return mainLibSymbol.get();
    auto address = llvm::RTDyldMemoryManager::getSymbolAddressInProcess(name);
    if (!address) {
      llvm::errs() << "Could not look up: " << name << '\n';
      return nullptr;
    }
    return llvm::JITSymbol(address, llvm::JITSymbolFlags::Exported);
  }

private:
  llvm::orc::ExecutionSession &session;
};
} // end anonymous namespace

namespace mlir {
namespace impl {
// Simple layered Orc JIT compilation engine.
class OrcJIT {
public:
  using IRTransformer = std::function<Error(llvm::Module *)>;

  // Construct a JIT engine for the target host defined by `machineBuilder`,
  // using the data layout provided as `dataLayout`.
  // Setup the object layer to use our custom memory manager in order to resolve
  // calls to library functions present in the process.
  OrcJIT(llvm::orc::JITTargetMachineBuilder machineBuilder,
         llvm::DataLayout layout, IRTransformer transform)
      : irTransformer(transform),
        objectLayer(
            session,
            [this]() { return llvm::make_unique<MemoryManager>(session); }),
        compileLayer(
            session, objectLayer,
            llvm::orc::ConcurrentIRCompiler(std::move(machineBuilder))),
        transformLayer(session, compileLayer, makeIRTransformFunction()),
        dataLayout(layout), mangler(session, this->dataLayout),
        threadSafeCtx(llvm::make_unique<llvm::LLVMContext>()) {
    session.getMainJITDylib().setGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            layout)));
  }

  // Create a JIT engine for the current host.
  static Expected<std::unique_ptr<OrcJIT>>
  createDefault(IRTransformer transformer) {
    auto machineBuilder = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!machineBuilder)
      return machineBuilder.takeError();

    auto dataLayout = machineBuilder->getDefaultDataLayoutForTarget();
    if (!dataLayout)
      return dataLayout.takeError();

    return llvm::make_unique<OrcJIT>(std::move(*machineBuilder),
                                     std::move(*dataLayout), transformer);
  }

  // Add an LLVM module to the main library managed by the JIT engine.
  Error addModule(std::unique_ptr<llvm::Module> M) {
    return transformLayer.add(
        session.getMainJITDylib(),
        llvm::orc::ThreadSafeModule(std::move(M), threadSafeCtx));
  }

  // Lookup a symbol in the main library managed by the JIT engine.
  Expected<llvm::JITEvaluatedSymbol> lookup(StringRef Name) {
    return session.lookup({&session.getMainJITDylib()}, mangler(Name.str()));
  }

private:
  // Wrap the `irTransformer` into a function that can be called by the
  // IRTranformLayer.  If `irTransformer` is not set up, return the module as is
  // without errors.
  llvm::orc::IRTransformLayer::TransformFunction makeIRTransformFunction() {
    return [this](llvm::orc::ThreadSafeModule module,
                  const llvm::orc::MaterializationResponsibility &resp)
               -> Expected<llvm::orc::ThreadSafeModule> {
      (void)resp;
      if (!irTransformer)
        return module;
      if (Error err = irTransformer(module.getModule()))
        return std::move(err);
      return module;
    };
  }

  IRTransformer irTransformer;
  llvm::orc::ExecutionSession session;
  llvm::orc::RTDyldObjectLinkingLayer objectLayer;
  llvm::orc::IRCompileLayer compileLayer;
  llvm::orc::IRTransformLayer transformLayer;
  llvm::DataLayout dataLayout;
  llvm::orc::MangleAndInterner mangler;
  llvm::orc::ThreadSafeContext threadSafeCtx;
};
} // end namespace impl
} // namespace mlir

// Wrap a string into an llvm::StringError.
static inline Error make_string_error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

// Given a list of PassRegistryEntry coming from a higher level, populates the
// given pass manager and appends the default set of required passes to lower to
// LLVMIR.
// Currently, these passes are:
// - constant folding
// - CSE
// - canonicalization
// - affine lowering
static void getDefaultPasses(
    PassManager &manager,
    const std::vector<const mlir::PassRegistryEntry *> &mlirPassRegistryList) {
  // Run each of the passes that were selected.
  for (const auto *passEntry : mlirPassRegistryList)
    passEntry->addToPipeline(manager);

  // Append the extra passes for lowering to MLIR.
  manager.addPass(mlir::createConstantFoldPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::createConvertToLLVMIRPass());
}

// Setup LLVM target triple from the current machine.
static bool setupTargetTriple(llvm::Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    llvm::errs() << "NO target: " << errorMessage << "\n";
    return true;
  }
  auto machine =
      target->createTargetMachine(targetTriple, "generic", "", {}, {});
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);
  return false;
}

static std::string makePackedFunctionName(StringRef name) {
  return "_mlir_" + name.str();
}

// For each function in the LLVM module, define an interface function that wraps
// all the arguments of the original function and all its results into an i8**
// pointer to provide a unified invocation interface.
void packFunctionArguments(llvm::Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  llvm::DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
        /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc =
        llvm::cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    llvm::SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto &indexedArg : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, indexedArg.index()));
      llvm::Value *argPtrPtr = builder.CreateGEP(argList, argIndex);
      llvm::Value *argPtr = builder.CreateLoad(argPtrPtr);
      argPtr = builder.CreateBitCast(
          argPtr, indexedArg.value().getType()->getPointerTo());
      llvm::Value *arg = builder.CreateLoad(argPtr);
      args.push_back(arg);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr = builder.CreateGEP(argList, retIndex);
      llvm::Value *retPtr = builder.CreateLoad(retPtrPtr);
      retPtr = builder.CreateBitCast(retPtr, result->getType()->getPointerTo());
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
}

// Out of line for PIMPL unique_ptr.
ExecutionEngine::~ExecutionEngine() = default;

std::unique_ptr<llvm::Module> translateModuleToLLVMIR(const Module &m);

Expected<std::unique_ptr<ExecutionEngine>> ExecutionEngine::create(
    Module *m, std::function<llvm::Error(llvm::Module *)> transformer) {
  auto engine = llvm::make_unique<ExecutionEngine>();
  auto expectedJIT = impl::OrcJIT::createDefault(transformer);
  if (!expectedJIT)
    return expectedJIT.takeError();

  // Construct and run the default MLIR pipeline.
  PassManager manager;
  getDefaultPasses(manager, {});
  if (failed(manager.run(m)))
    return make_string_error("passes failed");

  auto llvmModule = translateModuleToLLVMIR(*m);
  if (!llvmModule)
    return make_string_error("could not convert to LLVM IR");
  // FIXME: the triple should be passed to the translation or dialect conversion
  // instead of this.  Currently, the LLVM module created above has no triple
  // associated with it.
  setupTargetTriple(llvmModule.get());
  packFunctionArguments(llvmModule.get());

  if (auto err = (*expectedJIT)->addModule(std::move(llvmModule)))
    return std::move(err);
  engine->jit = std::move(*expectedJIT);

  return engine;
}

Expected<void (*)(void **)> ExecutionEngine::lookup(StringRef name) const {
  auto expectedSymbol = jit->lookup(makePackedFunctionName(name));
  if (!expectedSymbol)
    return expectedSymbol.takeError();
  auto rawFPtr = expectedSymbol->getAddress();
  auto fptr = reinterpret_cast<void (*)(void **)>(rawFPtr);
  if (!fptr)
    return make_string_error("looked up function is null");
  return fptr;
}

llvm::Error ExecutionEngine::invoke(StringRef name,
                                    MutableArrayRef<void *> args) {
  auto expectedFPtr = lookup(name);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  auto fptr = *expectedFPtr;

  (*fptr)(args.data());

  return llvm::Error::success();
}
