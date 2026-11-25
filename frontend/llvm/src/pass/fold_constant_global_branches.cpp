/*******************************************************************************
 *
 * \file
 * \brief Fold branches on effectively constant globals (v3.5.17)
 *
 * A constant integer global, or a writable one that is never written in the
 * module (internal linkage, or any linkage when the module defines `main`,
 * i.e. is a whole program), keeps its definitive initializer. A defined
 * function whose body is only `return <constant>` always returns that
 * constant.
 *
 * A conditional branch whose condition only depends on such values is replaced
 * by an unconditional branch to the taken successor. The value analysis
 * already proves the other edge unreachable; folding it also lets the
 * syntactic LLVM proofs (e.g. loop initialization, initialized promotion)
 * rely on the real control flow of patterns such as
 *
 *   static int staticTrue = 1;          if (staticTrue) { ... }
 *   const int GLOBAL_CONST_TRUE = 1;    if (GLOBAL_CONST_TRUE) { ... }
 *   int globalReturnsTrue() { return 1; }  if (globalReturnsTrue()) { ... }
 *
 * Calls are kept (only the branch is folded), so no diagnostic of the callee
 * or of the call itself is removed. Unsupported shapes are left unchanged.
 *
 ******************************************************************************/

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Optional.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>

#include <ikos/frontend/llvm/pass.hpp>

namespace {

class FoldConstantGlobalBranchesPass final : public llvm::ModulePass {
public:
  static char ID;

  FoldConstantGlobalBranchesPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage&) const override {}

private:
  llvm::DenseMap< const llvm::GlobalVariable*, llvm::ConstantInt* > _globals;
  llvm::DenseMap< const llvm::Function*, llvm::ConstantInt* > _functions;

  /// \brief Return true if every (transitive) use only reads the global
  static bool only_read(const llvm::Value* value, unsigned depth = 0) {
    if (depth > 4) {
      return false;
    }
    for (const llvm::User* user : value->users()) {
      if (auto load = llvm::dyn_cast< llvm::LoadInst >(user)) {
        if (!load->isSimple() || load->getPointerOperand() != value) {
          return false;
        }
      } else if (auto expr = llvm::dyn_cast< llvm::ConstantExpr >(user)) {
        if (expr->getOpcode() != llvm::Instruction::BitCast ||
            !only_read(expr, depth + 1)) {
          return false;
        }
      } else {
        // Stores, calls, comparisons, escapes into other globals, etc.
        return false;
      }
    }
    return true;
  }

  void collect_globals(llvm::Module& module) {
    // A writable global with external linkage may be written by another
    // translation unit, unless the module is a whole program.
    const llvm::Function* main = module.getFunction("main");
    const bool whole_program = main != nullptr && !main->isDeclaration();
    for (llvm::GlobalVariable& gv : module.globals()) {
      if (!gv.hasDefinitiveInitializer() ||
          !gv.getValueType()->isIntegerTy() || gv.isThreadLocal()) {
        continue;
      }
      auto init = llvm::dyn_cast< llvm::ConstantInt >(gv.getInitializer());
      if (init == nullptr) {
        continue;
      }
      if (gv.isConstant() ||
          ((gv.hasLocalLinkage() || whole_program) && only_read(&gv))) {
        this->_globals[&gv] = init;
      }
    }
  }

  void collect_functions(llvm::Module& module) {
    for (llvm::Function& fun : module) {
      if (fun.isDeclaration() || fun.isVarArg() || fun.size() != 1 ||
          !fun.getReturnType()->isIntegerTy() || fun.isInterposable()) {
        continue;
      }
      llvm::ConstantInt* result = nullptr;
      bool pure = true;
      for (llvm::Instruction& inst : fun.front()) {
        if (llvm::isa< llvm::DbgInfoIntrinsic >(&inst)) {
          continue;
        }
        if (auto ret = llvm::dyn_cast< llvm::ReturnInst >(&inst)) {
          result = llvm::dyn_cast_or_null< llvm::ConstantInt >(
              ret->getReturnValue());
          continue;
        }
        pure = false;
        break;
      }
      if (pure && result != nullptr) {
        this->_functions[&fun] = result;
      }
    }
  }

  /// \brief Evaluate an integer condition built from known constants
  llvm::ConstantInt* evaluate(llvm::Value* value, unsigned depth = 0) {
    if (depth > 8) {
      return nullptr;
    }
    if (auto cst = llvm::dyn_cast< llvm::ConstantInt >(value)) {
      return cst;
    }
    if (auto load = llvm::dyn_cast< llvm::LoadInst >(value)) {
      auto gv = llvm::dyn_cast< llvm::GlobalVariable >(
          load->getPointerOperand()->stripPointerCasts());
      if (gv == nullptr || !load->isSimple()) {
        return nullptr;
      }
      auto it = this->_globals.find(gv);
      if (it == this->_globals.end() ||
          it->second->getType() != load->getType()) {
        return nullptr;
      }
      return it->second;
    }
    if (auto call = llvm::dyn_cast< llvm::CallInst >(value)) {
      auto callee = call->getCalledFunction();
      if (callee == nullptr || call->arg_size() != 0) {
        return nullptr;
      }
      auto it = this->_functions.find(callee);
      if (it == this->_functions.end() ||
          it->second->getType() != call->getType()) {
        return nullptr;
      }
      return it->second;
    }
    if (auto cast = llvm::dyn_cast< llvm::CastInst >(value)) {
      if (!cast->getType()->isIntegerTy() ||
          !cast->getOperand(0)->getType()->isIntegerTy()) {
        return nullptr;
      }
      auto operand = this->evaluate(cast->getOperand(0), depth + 1);
      if (operand == nullptr) {
        return nullptr;
      }
      auto folded = llvm::ConstantExpr::getCast(cast->getOpcode(),
                                                operand,
                                                cast->getType());
      return llvm::dyn_cast< llvm::ConstantInt >(folded);
    }
    if (auto cmp = llvm::dyn_cast< llvm::ICmpInst >(value)) {
      auto left = this->evaluate(cmp->getOperand(0), depth + 1);
      auto right = this->evaluate(cmp->getOperand(1), depth + 1);
      if (left == nullptr || right == nullptr) {
        return nullptr;
      }
      return llvm::dyn_cast< llvm::ConstantInt >(
          llvm::ConstantExpr::getICmp(
              static_cast< unsigned short >(cmp->getPredicate()), left, right));
    }
    return nullptr;
  }

public:
  bool runOnModule(llvm::Module& module) override {
    this->collect_globals(module);
    this->collect_functions(module);
    if (this->_globals.empty() && this->_functions.empty()) {
      return false;
    }
    bool changed = false;
    for (llvm::Function& fun : module) {
      for (llvm::BasicBlock& block : fun) {
        auto branch = llvm::dyn_cast< llvm::BranchInst >(block.getTerminator());
        if (branch == nullptr || !branch->isConditional() ||
            branch->getSuccessor(0) == branch->getSuccessor(1)) {
          continue;
        }
        auto condition = this->evaluate(branch->getCondition());
        if (condition == nullptr || condition->getBitWidth() != 1) {
          continue;
        }
        llvm::BasicBlock* taken = branch->getSuccessor(condition->isOne() ? 0 : 1);
        llvm::BasicBlock* dead = branch->getSuccessor(condition->isOne() ? 1 : 0);
        dead->removePredecessor(&block);
        auto replacement = llvm::BranchInst::Create(taken, branch);
        replacement->setDebugLoc(branch->getDebugLoc());
        branch->eraseFromParent();
        changed = true;
      }
    }
    return changed;
  }
};

} // namespace

char FoldConstantGlobalBranchesPass::ID = 0;

// LLVM pass registration macros require names from the llvm namespace.
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace llvm;

INITIALIZE_PASS(FoldConstantGlobalBranchesPass,
                "fold-constant-global-branches",
                "Fold branches on effectively constant globals",
                false,
                false);

llvm::ModulePass* ikos::frontend::pass::
    create_fold_constant_global_branches_pass() {
  return new FoldConstantGlobalBranchesPass();
}
