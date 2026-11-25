/*******************************************************************************
 *
 * \file
 * \brief Separate proven single-iteration loops without folding body operations
 *
 * Contributors: IKOS developers
 *
 * Contact: ikos@lists.nasa.gov
 *
 * Notices:
 *
 * Copyright (c) 2011-2019 United States Government as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 * All Rights Reserved.
 *
 * Disclaimers:
 *
 * No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
 * ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
 * TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS,
 * ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
 * OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE
 * ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO
 * THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
 * ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
 * RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
 * RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
 * DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE,
 * IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
 *
 * Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST
 * THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL
 * AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS
 * IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH
 * USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM,
 * RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD
 * HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS,
 * AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.
 * RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE,
 * UNILATERAL TERMINATION OF THIS AGREEMENT.
 *
 ******************************************************************************/

#include <utility>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Pass.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <ikos/frontend/llvm/pass.hpp>

namespace {

/// Separate the initial and final tests of a proven single-iteration loop.
/// Keep every body instruction, including uninitialized reads and side effects.
/// General LLVM loop optimizations may fold undef and hide those diagnostics.
struct SimplifySingleIterationLoopsPass final : public llvm::FunctionPass {
  static char ID;

  SimplifySingleIterationLoopsPass() : FunctionPass(ID) {}

  static llvm::Value* add_one_operand(llvm::Value* value) {
    auto* add = llvm::dyn_cast< llvm::BinaryOperator >(value);
    if (!add || add->getOpcode() != llvm::Instruction::Add) {
      return nullptr;
    }
    for (unsigned i = 0; i < 2; ++i) {
      auto* one = llvm::dyn_cast< llvm::ConstantInt >(add->getOperand(i));
      if (one && one->isOne()) {
        return add->getOperand(1 - i);
      }
    }
    return nullptr;
  }

  static bool memory_counter(llvm::Value* induction,
                             llvm::BasicBlock* header,
                             llvm::BasicBlock* preheader,
                             llvm::BasicBlock* latch,
                             llvm::DominatorTree& dominators) {
    auto* load = llvm::dyn_cast< llvm::LoadInst >(induction);
    auto* slot =
        load ? llvm::dyn_cast< llvm::AllocaInst >(load->getPointerOperand())
             : nullptr;
    if (!load || !load->isSimple() || load->getParent() != header || !slot ||
        !slot->isStaticAlloca() || slot->isArrayAllocation() ||
        !slot->getAllocatedType()->isIntegerTy()) {
      return false;
    }
    llvm::StoreInst* initial = nullptr;
    llvm::StoreInst* increment = nullptr;
    for (llvm::User* user : slot->users()) {
      if (auto* read = llvm::dyn_cast< llvm::LoadInst >(user)) {
        if (!read->isSimple() || read->getPointerOperand() != slot) {
          return false;
        }
      } else if (auto* store = llvm::dyn_cast< llvm::StoreInst >(user)) {
        if (!store->isSimple() || store->getPointerOperand() != slot) {
          return false;
        }
        if (store->getParent() == preheader && !initial) {
          initial = store;
        } else if (store->getParent() == latch && !increment) {
          increment = store;
        } else {
          return false;
        }
      } else {
        // In particular, do not infer a counter value through an escaped slot.
        return false;
      }
    }
    auto* zero =
        initial
            ? llvm::dyn_cast< llvm::ConstantInt >(initial->getValueOperand())
            : nullptr;
    auto* old = increment ? llvm::dyn_cast_or_null< llvm::LoadInst >(
                                add_one_operand(increment->getValueOperand()))
                          : nullptr;
    if (!zero || !zero->isZero() || !old || !old->isSimple() ||
        old->getPointerOperand() != slot || old->getParent() != latch ||
        !dominators.dominates(old, increment)) {
      return false;
    }
    for (llvm::User* user : slot->users()) {
      if (auto* read = llvm::dyn_cast< llvm::LoadInst >(user)) {
        if (!dominators.dominates(initial, read)) {
          return false;
        }
      }
    }
    return true;
  }

  /// \brief Collect the targets of the exits other than the header test
  ///
  /// v3.5.15.1: an exit to a block ending in `unreachable` (e.g. the
  /// `exit(-1)` of `if (p == NULL) { exit(-1); }`) never comes back, so it is
  /// allowed. Any other extra exit is rejected.
  static bool noreturn_exits(
      llvm::Loop* loop,
      llvm::BasicBlock* header,
      llvm::SmallPtrSetImpl< llvm::BasicBlock* >& targets) {
    llvm::SmallVector< llvm::Loop::Edge, 4 > edges;
    loop->getExitEdges(edges);
    for (const auto& edge : edges) {
      const llvm::BasicBlock* from = edge.first;
      llvm::BasicBlock* to = const_cast< llvm::BasicBlock* >(edge.second);
      if (from == header) {
        continue;
      }
      if (!llvm::isa< llvm::UnreachableInst >(to->getTerminator()) ||
          to->hasAddressTaken() || to->getSinglePredecessor() != from) {
        return false;
      }
      targets.insert(to);
    }
    return true;
  }

  static bool simplify(llvm::Loop* loop, llvm::DominatorTree& dominators) {
    auto* header = loop->getHeader();
    auto* preheader = loop->getLoopPreheader();
    auto* latch = loop->getLoopLatch();
    auto* branch = llvm::dyn_cast< llvm::BranchInst >(header->getTerminator());
    // v3.5.15.1: sub-loops are allowed. Only the header is cloned and the
    // back edge redirected; body blocks (including inner loops) are kept as
    // is and run once. An inner exit leaving this loop makes it multi-exit.
    llvm::SmallPtrSet< llvm::BasicBlock*, 4 > noreturn;
    if (!preheader || !latch || header == latch || header->hasAddressTaken() ||
        !noreturn_exits(loop, header, noreturn) ||
        !branch || !branch->isConditional() ||
        !loop->contains(branch->getSuccessor(0)) ||
        loop->contains(branch->getSuccessor(1))) {
      return false;
    }
    auto* backedge = llvm::dyn_cast< llvm::BranchInst >(latch->getTerminator());
    if (!backedge || !backedge->isUnconditional() ||
        backedge->getSuccessor(0) != header) {
      return false;
    }
    auto* cmp = llvm::dyn_cast< llvm::ICmpInst >(branch->getCondition());
    if (!cmp || cmp->getParent() != header ||
        (cmp->getPredicate() != llvm::ICmpInst::ICMP_SLT &&
         cmp->getPredicate() != llvm::ICmpInst::ICMP_ULT)) {
      return false;
    }
    auto* bound = llvm::dyn_cast< llvm::ConstantInt >(cmp->getOperand(1));
    if (!bound || !bound->isOne() || (cmp->isSigned() && bound->isNegative())) {
      return false;
    }
    llvm::Value* induction = cmp->getOperand(0);
    auto* phi = llvm::dyn_cast< llvm::PHINode >(induction);
    if (phi) {
      if (phi->getParent() != header || phi->getNumIncomingValues() != 2 ||
          phi->getBasicBlockIndex(preheader) < 0 ||
          phi->getBasicBlockIndex(latch) < 0) {
        return false;
      }
      auto* zero = llvm::dyn_cast< llvm::ConstantInt >(
          phi->getIncomingValueForBlock(preheader));
      if (!zero || !zero->isZero() ||
          add_one_operand(phi->getIncomingValueForBlock(latch)) != phi) {
        return false;
      }
    } else if (!memory_counter(induction,
                               header,
                               preheader,
                               latch,
                               dominators)) {
      return false;
    }
    // The header is duplicated to preserve its final values for exit users.
    // Reject other computations, loads or effects rather than speculate them.
    for (llvm::Instruction& inst : *header) {
      if (auto* carried = llvm::dyn_cast< llvm::PHINode >(&inst)) {
        if (carried->getNumIncomingValues() != 2 ||
            carried->getBasicBlockIndex(preheader) < 0 ||
            carried->getBasicBlockIndex(latch) < 0) {
          return false;
        }
      } else if (&inst != induction && &inst != cmp && &inst != branch &&
                 !llvm::isa< llvm::DbgInfoIntrinsic >(inst)) {
        return false;
      }
    }

    llvm::BasicBlock* body = branch->getSuccessor(0);
    llvm::BasicBlock* exit = branch->getSuccessor(1);
    llvm::ValueToValueMapTy values;
    llvm::BasicBlock* final =
        llvm::CloneBasicBlock(header, values, ".last", header->getParent());
    final->moveBefore(exit);
    for (llvm::PHINode& carried : header->phis()) {
      auto* copy = llvm::cast< llvm::PHINode >(values[&carried]);
      values[&carried] = carried.getIncomingValueForBlock(latch);
      copy->eraseFromParent();
    }
    for (llvm::Instruction& inst : *final) {
      llvm::RemapInstruction(&inst,
                             values,
                             llvm::RF_NoModuleLevelChanges |
                                 llvm::RF_IgnoreMissingLocals);
    }

    // A header-defined value used after the loop denotes the final test,
    // not the initial test. This includes non-LCSSA users of the induction.
    // Snapshot all replacements first: a carried value can itself be another
    // header PHI's first-iteration value. Do not remap that new use twice.
    llvm::SmallVector< std::pair< llvm::Use*, llvm::Value* >, 16 > replacements;
    for (llvm::Instruction& inst : *header) {
      if (inst.isTerminator() || inst.getType()->isVoidTy()) {
        continue;
      }
      for (llvm::Use& use : inst.uses()) {
        auto* user = llvm::dyn_cast< llvm::Instruction >(use.getUser());
        // A noreturn exit runs during the single iteration: keep the
        // first-iteration value there (v3.5.15.1).
        if (user && !loop->contains(user->getParent()) &&
            user->getParent() != final && noreturn.count(user->getParent()) == 0) {
          replacements.emplace_back(&use, values[&inst]);
        }
      }
    }
    for (const auto& replacement : replacements) {
      replacement.first->set(replacement.second);
    }
    for (llvm::PHINode& exit_phi : exit->phis()) {
      int index = exit_phi.getBasicBlockIndex(header);
      if (index >= 0) {
        exit_phi.setIncomingBlock(static_cast< unsigned >(index), final);
      }
    }
    for (llvm::PHINode& carried : header->phis()) {
      carried.removeIncomingValue(latch, false);
    }
    backedge->setSuccessor(0, final);
    auto* initial_branch = llvm::BranchInst::Create(body, branch);
    initial_branch->setDebugLoc(branch->getDebugLoc());
    branch->eraseFromParent();
    auto* final_branch = llvm::BranchInst::Create(exit, final->getTerminator());
    final_branch->setDebugLoc(final->getTerminator()->getDebugLoc());
    final->getTerminator()->eraseFromParent();
    return true;
  }

  bool runOnFunction(llvm::Function& function) override {
    if (function.isDeclaration()) {
      return false;
    }
    unsigned instructions = 0;
    for (llvm::BasicBlock& block : function) {
      for (llvm::Instruction& inst : block) {
        if (++instructions > 32768) {
          return false;
        }
        if (auto* call = llvm::dyn_cast< llvm::CallBase >(&inst)) {
          if (call->isInlineAsm() ||
              call->hasFnAttr(llvm::Attribute::ReturnsTwice)) {
            return false;
          }
        }
        if (auto* intrinsic = llvm::dyn_cast< llvm::IntrinsicInst >(&inst)) {
          if (intrinsic->getIntrinsicID() == llvm::Intrinsic::stacksave ||
              intrinsic->getIntrinsicID() == llvm::Intrinsic::stackrestore) {
            return false;
          }
        }
      }
    }
    bool changed = false;
    for (unsigned count = 0; count < 128; ++count) {
      // Rebuild after each CFG change; never use stale loop/dominator facts.
      llvm::DominatorTree dominators(function);
      llvm::LoopInfo loops(dominators);
      llvm::SmallVector< llvm::Loop*, 8 > pending(loops.begin(), loops.end());
      bool simplified = false;
      while (!pending.empty()) {
        llvm::Loop* loop = pending.pop_back_val();
        if (simplify(loop, dominators)) {
          simplified = true;
          changed = true;
          break;
        }
        for (llvm::Loop* child : loop->getSubLoops()) {
          pending.push_back(child);
        }
      }
      if (!simplified) {
        break;
      }
    }
    return changed;
  }
};

} // namespace

char SimplifySingleIterationLoopsPass::ID = 0;

// LLVM pass registration macros require names from the llvm namespace.
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace llvm;

INITIALIZE_PASS(SimplifySingleIterationLoopsPass,
                "simplify-single-iteration-loops",
                "Separate proven single-iteration loop tests",
                false,
                false);

llvm::FunctionPass* ikos::frontend::pass::
    create_simplify_single_iteration_loops_pass() {
  return new SimplifySingleIterationLoopsPass();
}
