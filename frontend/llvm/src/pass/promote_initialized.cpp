/*******************************************************************************
 *
 * \file
 * \brief Promote initialized stack slots without erasing uninitialized reads
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

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Pass.h>
#include <llvm/Transforms/Utils/PromoteMemToReg.h>

#include <ikos/frontend/llvm/pass.hpp>

namespace {

/// Keep potentially uninitialized reads in memory. LLVM's ordinary mem2reg
/// may replace a phi of undef and an initialized value with the latter value,
/// which is valid for optimization but hides the uninitialized path from UVA.
struct PromoteInitializedPass final : public llvm::FunctionPass {
  static char ID;

  PromoteInitializedPass() : FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage& usage) const override {
    usage.setPreservesCFG();
  }

  static bool has_store_before_loads(llvm::AllocaInst* slot) {
    // Explicit undef/poison stores must not be folded into another incoming
    // value either. Ordinary stores of SSA values retain their initialization
    // information when promoted and are safe for this transformation.
    for (llvm::User* user : slot->users()) {
      if (auto* store = llvm::dyn_cast< llvm::StoreInst >(user)) {
        if (llvm::isa< llvm::UndefValue >(store->getValueOperand())) {
          return false;
        }
      }
    }

    // Explore paths on which no store has occurred yet. A store ends such a
    // path; any load encountered before it prevents promotion. Visiting each
    // block once handles loops without assuming they execute at least once.
    // Infeasible CFG paths are conservatively included; promotion is only
    // an optimization, so failure to prove this property keeps the slot.
    llvm::SmallVector< llvm::BasicBlock*, 16 > worklist;
    llvm::SmallPtrSet< llvm::BasicBlock*, 32 > visited;
    worklist.push_back(&slot->getFunction()->getEntryBlock());
    while (!worklist.empty()) {
      llvm::BasicBlock* block = worklist.pop_back_val();
      if (!visited.insert(block).second) {
        continue;
      }
      bool stored = false;
      for (llvm::Instruction& inst : *block) {
        if (auto* load = llvm::dyn_cast< llvm::LoadInst >(&inst)) {
          if (load->getPointerOperand() == slot) {
            return false;
          }
        } else if (auto* store = llvm::dyn_cast< llvm::StoreInst >(&inst)) {
          if (store->getPointerOperand() == slot) {
            stored = true;
            break;
          }
        }
      }
      if (!stored) {
        for (llvm::BasicBlock* successor : llvm::successors(block)) {
          worklist.push_back(successor);
        }
      }
    }
    return true;
  }

  bool runOnFunction(llvm::Function& function) override {
    if (this->skipFunction(function)) {
      return false;
    }
    llvm::DominatorTree dominators(function);
    llvm::AssumptionCache assumptions(function);
    bool changed = false;
    for (;;) {
      llvm::SmallVector< llvm::AllocaInst*, 16 > slots;
      for (llvm::Instruction& inst : function.getEntryBlock()) {
        auto* slot = llvm::dyn_cast< llvm::AllocaInst >(&inst);
        if (slot && llvm::isAllocaPromotable(slot) &&
            has_store_before_loads(slot)) {
          slots.push_back(slot);
        }
      }
      if (slots.empty()) {
        return changed;
      }
      llvm::PromoteMemToReg(slots, dominators, &assumptions);
      changed = true;
      // Promoting pointer slots can expose further promotable allocas.
    }
  }
};

} // end anonymous namespace

char PromoteInitializedPass::ID = 0;

// LLVM pass registration macros require names from the llvm namespace.
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace llvm;

INITIALIZE_PASS(PromoteInitializedPass,
                "promote-initialized",
                "Promote stack slots while preserving uninitialized reads",
                false,
                false);

llvm::FunctionPass* ikos::frontend::pass::create_promote_initialized_pass() {
  return new PromoteInitializedPass();
}
