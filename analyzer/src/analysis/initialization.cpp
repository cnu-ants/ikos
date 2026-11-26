#include <cstdint>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>

#include <ikos/analyzer/analysis/call_context.hpp>
#include <ikos/analyzer/analysis/initialization.hpp>
#include <ikos/ar/semantic/statement.hpp>

namespace ikos {
namespace analyzer {
namespace {

/// A bounded, backward must-initialization proof. Unlike the nonrelational
/// memory join, this keeps the correlation between a pointer PHI and the
/// store on each incoming edge. It follows direct calls in their own frames,
/// but does not infer numeric values or summarize loops.
class InitializationProof {
private:
  struct Address {
    llvm::Value* base;
    int64_t offset;
    // A nonescaping pointer slot denotes its contents at the scan position.
    bool indirect;
  };

  llvm::Function& _function;
  const llvm::DataLayout& _layout;
  llvm::DominatorTree _dominators;
  llvm::SmallPtrSet< llvm::BasicBlock*, 16 > _active;
  llvm::SmallPtrSet< llvm::Function*, 8 > _active_calls;
  unsigned& _budget;
  unsigned _call_depth;
  llvm::CallBase* _invocation;
  InitializationProof* _caller;
  CallContext* _caller_context = nullptr;
  bool _nonnull_result;
  bool _prove_initialized;

public:
  InitializationProof(llvm::Function& function,
                      CallContext* context,
                      unsigned& budget,
                      bool prove_initialized,
                      unsigned call_depth = 0,
                      InitializationProof* caller = nullptr,
                      llvm::CallBase* invocation = nullptr,
                      bool nonnull_result = false)
      : _function(function),
        _layout(function.getParent()->getDataLayout()),
        _dominators(function),
        _budget(budget),
        _call_depth(call_depth),
        _invocation(invocation),
        _caller(caller),
        _nonnull_result(nonnull_result),
        _prove_initialized(prove_initialized) {
    if (!invocation && context && !context->empty() &&
        context->call()->has_frontend()) {
      auto call = llvm::dyn_cast< llvm::CallBase >(
          context->call()->frontend< llvm::Value >());
      // Only a direct call with an exact callee match supplies bindings.
      // Empty/merged contexts and unresolved indirect calls remain unknown.
      if (call && call->getCalledFunction() == &function) {
        this->_invocation = call;
        this->_caller_context = context->parent();
      }
    }
  }

private:
  bool consume() {
    if (this->_budget == 0) {
      return false;
    }
    --this->_budget;
    return true;
  }

  template < typename Query >
  bool in_caller(Query query) {
    if (!this->_invocation || !this->consume() || this->_call_depth >= 8) {
      return false;
    }
    if (this->_caller) {
      // Resume before the call in a fresh CFG walk. Reusing the suspended
      // caller's active-block set would incorrectly reject pointer-return
      // forwarding within the same block as a cycle.
      InitializationProof caller(this->_caller->_function,
                                 nullptr,
                                 this->_budget,
                                 this->_prove_initialized,
                                 this->_call_depth + 1,
                                 this->_caller->_caller,
                                 this->_caller->_invocation,
                                 this->_caller->_nonnull_result);
      caller._caller_context = this->_caller->_caller_context;
      return query(caller);
    }
    InitializationProof caller(*this->_invocation->getFunction(),
                               this->_caller_context,
                               this->_budget,
                               this->_prove_initialized,
                               this->_call_depth + 1);
    return query(caller);
  }

  // Restrict spilled pointer reasoning to an ordinary, unaliased stack slot.
  // In particular, unknown calls cannot modify this slot through an escape.
  bool pointer_slot(llvm::AllocaInst* slot) {
    if (!slot->getAllocatedType()->isPointerTy() || slot->isArrayAllocation()) {
      return false;
    }
    for (llvm::User* user : slot->users()) {
      if (!this->consume()) {
        return false;
      }
      if (auto load = llvm::dyn_cast< llvm::LoadInst >(user)) {
        if (!load->isSimple() || load->getPointerOperand() != slot) {
          return false;
        }
      } else if (auto store = llvm::dyn_cast< llvm::StoreInst >(user)) {
        if (!store->isSimple() || store->getPointerOperand() != slot ||
            store->getValueOperand() == slot) {
          return false;
        }
      } else if (!llvm::isa< llvm::DbgInfoIntrinsic >(user)) {
        return false;
      }
    }
    return true;
  }

  // Two reloads of a slot denote the same pointer only if the slot cannot
  // have been assigned between the earlier reload and its use.
  bool stable(llvm::LoadInst* load,
              llvm::Instruction* use,
              llvm::AllocaInst* slot) {
    for (llvm::User* user : slot->users()) {
      if (!this->consume()) {
        return false;
      }
      auto store = llvm::dyn_cast< llvm::StoreInst >(user);
      if (store &&
          llvm::isPotentiallyReachable(load,
                                       store,
                                       nullptr,
                                       &this->_dominators) &&
          llvm::isPotentiallyReachable(store,
                                       use,
                                       nullptr,
                                       &this->_dominators)) {
        return false;
      }
    }
    return true;
  }

  bool address(llvm::Value* value, llvm::Instruction* use, Address& result) {
    if (!this->consume() || !value->getType()->isPointerTy()) {
      return false;
    }
    unsigned bits = this->_layout.getIndexTypeSizeInBits(value->getType());
    if (bits > 64) {
      return false;
    }
    llvm::APInt offset(bits, 0);
    llvm::Value* base =
        value->stripAndAccumulateConstantOffsets(this->_layout, offset, true);
    result = {base, offset.getSExtValue(), false};
    if (auto load = llvm::dyn_cast< llvm::LoadInst >(base)) {
      auto slot = llvm::dyn_cast< llvm::AllocaInst >(load->getPointerOperand());
      if (slot && load->isSimple() && this->pointer_slot(slot) &&
          this->stable(load, use, slot)) {
        result.base = slot;
        result.indirect = true;
      }
    }
    return true;
  }

  bool initialized(llvm::Value* value, unsigned depth = 0) {
    // A store of undef/poison or a loaded, potentially uninitialized value
    // must not discharge the obligation. More elaborate values remain the
    // responsibility of the ordinary abstract interpretation.
    if (depth >= 32 || !this->consume()) {
      return false;
    }
    if (llvm::isa< llvm::ConstantInt >(value) ||
        llvm::isa< llvm::ConstantFP >(value) ||
        llvm::isa< llvm::ConstantPointerNull >(value)) {
      return true;
    }
    if (auto arg = llvm::dyn_cast< llvm::Argument >(value)) {
      if (arg->getParent() != &this->_function || !this->_invocation ||
          arg->getArgNo() >= this->_invocation->arg_size()) {
        return false;
      }
      return this->in_caller([&](InitializationProof& caller) {
        return caller.initialized(this->_invocation->getArgOperand(
                                      arg->getArgNo()),
                                  depth + 1);
      });
    }
    if (auto call = llvm::dyn_cast< llvm::CallBase >(value)) {
      auto callee = call->getCalledFunction();
      return callee && callee->isDeclaration() && callee->getName() == "rand" &&
             call->arg_size() == 0 && call->getType()->isIntegerTy(32);
    }
    auto load = llvm::dyn_cast< llvm::LoadInst >(value);
    auto slot =
        load ? llvm::dyn_cast< llvm::AllocaInst >(load->getPointerOperand())
             : nullptr;
    if (!slot || !load->isSimple() || slot->isArrayAllocation() ||
        slot->getAllocatedType() != load->getType()) {
      return false;
    }
    // At opt=none a formal scalar is usually spilled. Only forward an
    // ordinary, nonescaping slot with one dominating store. Do not assume
    // that parameters are initialized just because LLVM marks them noundef.
    llvm::StoreInst* definition = nullptr;
    for (llvm::User* user : slot->users()) {
      if (!this->consume()) {
        return false;
      }
      if (auto read = llvm::dyn_cast< llvm::LoadInst >(user)) {
        if (!read->isSimple() || read->getPointerOperand() != slot) {
          return false;
        }
      } else if (auto store = llvm::dyn_cast< llvm::StoreInst >(user)) {
        if (definition || !store->isSimple() ||
            store->getPointerOperand() != slot ||
            store->getValueOperand() == slot) {
          return false;
        }
        definition = store;
      } else if (!llvm::isa< llvm::DbgInfoIntrinsic >(user)) {
        return false;
      }
    }
    return definition && this->_dominators.dominates(definition, load) &&
           this->initialized(definition->getValueOperand(), depth + 1);
  }

  bool excluded_null(const Address& read) const {
    // A null result cannot be the source of a valid memory read. This is
    // used only for allocation-wrapper return paths; nullity/bounds checks
    // still diagnose invalid accesses separately.
    auto null = llvm::dyn_cast< llvm::ConstantPointerNull >(read.base);
    return this->_nonnull_result && !read.indirect && read.offset == 0 &&
           null && null->getType()->getAddressSpace() == 0 &&
           !this->_function.hasFnAttribute(llvm::Attribute::NullPointerIsValid);
  }

  bool returned_memory(llvm::CallBase* call,
                       const Address& read,
                       uint64_t size,
                       unsigned depth) {
    auto callee = call->getCalledFunction();
    if (!callee || callee->isDeclaration() || this->_call_depth >= 8 ||
        read.indirect || !this->consume()) {
      return false;
    }
    InitializationProof child(*callee,
                              nullptr,
                              this->_budget,
                              this->_prove_initialized,
                              this->_call_depth + 1,
                              this,
                              call,
                              true);
    bool has_return = false;
    for (llvm::BasicBlock& block : *callee) {
      if (!this->consume()) {
        return false;
      }
      auto ret = llvm::dyn_cast< llvm::ReturnInst >(block.getTerminator());
      if (!ret) {
        continue;
      }
      has_return = true;
      auto value = ret->getReturnValue();
      Address returned;
      int64_t offset;
      if (!value || !child.address(value, ret, returned) ||
          __builtin_add_overflow(returned.offset, read.offset, &offset)) {
        return false;
      }
      returned.offset = offset;
      if (!child.scan(&block, ret->getIterator(), returned, size, depth + 1)) {
        return false;
      }
    }
    return has_return;
  }

  bool argument_memory(const Address& read, uint64_t size, unsigned depth) {
    auto arg = llvm::dyn_cast< llvm::Argument >(read.base);
    if (read.indirect || !arg || arg->getParent() != &this->_function ||
        !this->_invocation ||
        arg->getArgNo() >= this->_invocation->arg_size()) {
      return false;
    }
    return this->in_caller([&](InitializationProof& caller) {
      Address actual;
      int64_t offset;
      auto call = this->_invocation;
      if (!caller.address(call->getArgOperand(arg->getArgNo()), call, actual) ||
          __builtin_add_overflow(actual.offset, read.offset, &offset)) {
        return false;
      }
      actual.offset = offset;
      return caller.scan(call->getParent(),
                         call->getIterator(),
                         actual,
                         size,
                         depth + 1);
    });
  }

  bool preserves_memory(llvm::CallBase* call, unsigned depth = 0) {
    if (!this->consume() || depth >= 32) {
      return false;
    }
    if (llvm::isa< llvm::DbgInfoIntrinsic >(call) ||
        call->doesNotAccessMemory() || call->onlyReadsMemory()) {
      return true;
    }
    // v3.5.21: a call to a function declared without prototype goes through
    // a bitcast on x86_64 (e.g. `int globalReturnsTrueOrFalse();`). Look
    // through the cast: only the callee body is inspected here, so the call
    // signature does not matter (argument bindings are not used).
    auto callee = llvm::dyn_cast< llvm::Function >(
        call->getCalledOperand()->stripPointerCasts());
    if (!callee) {
      return false;
    }
    // Match IKOS's existing libc rand model, including through wrappers.
    if (callee->isDeclaration()) {
      return callee->getName() == "rand" && call->arg_size() == 0 &&
             call->getType()->isIntegerTy(32);
    }
    if (!this->_active_calls.insert(callee).second) {
      return false;
    }
    bool result = true;
    for (llvm::BasicBlock& block : *callee) {
      for (llvm::Instruction& inst : block) {
        if (!this->consume()) {
          result = false;
          break;
        }
        if (auto nested = llvm::dyn_cast< llvm::CallBase >(&inst)) {
          result = this->preserves_memory(nested, depth + 1);
        } else if (inst.mayWriteToMemory()) {
          result = false;
        }
        if (!result) {
          break;
        }
      }
      if (!result) {
        break;
      }
    }
    this->_active_calls.erase(callee);
    return result;
  }

  bool initialized_allocation(llvm::Instruction& inst,
                              const Address& read,
                              uint64_t size) const {
    auto call = llvm::dyn_cast< llvm::CallBase >(&inst);
    auto callee = call ? call->getCalledFunction() : nullptr;
    if (!callee || !callee->isDeclaration() || callee->getName() != "calloc" ||
        call->arg_size() != 2 || read.indirect || read.offset < 0) {
      return false;
    }
    auto count = llvm::dyn_cast< llvm::ConstantInt >(call->getArgOperand(0));
    auto element = llvm::dyn_cast< llvm::ConstantInt >(call->getArgOperand(1));
    if (!count || !element || count->getBitWidth() != element->getBitWidth() ||
        count->getBitWidth() > 64 ||
        count->getBitWidth() != this->_layout.getPointerSizeInBits()) {
      return false;
    }
    bool overflow;
    auto product = count->getValue().umul_ov(element->getValue(), overflow);
    uint64_t offset = static_cast< uint64_t >(read.offset);
    return !overflow && offset <= product.getZExtValue() &&
           size <= product.getZExtValue() - offset;
  }

  bool untouched_allocation(llvm::Instruction& inst,
                            const Address& read,
                            uint64_t size) const {
    auto call = llvm::dyn_cast< llvm::CallBase >(&inst);
    auto callee = call ? call->getCalledFunction() : nullptr;
    if (!callee || !callee->isDeclaration() || callee->getName() != "malloc" ||
        call->arg_size() != 1 || read.indirect || read.offset < 0) {
      return false;
    }
    auto allocated =
        llvm::dyn_cast< llvm::ConstantInt >(call->getArgOperand(0));
    if (!allocated || allocated->getBitWidth() > 64 ||
        allocated->getBitWidth() != this->_layout.getPointerSizeInBits()) {
      return false;
    }
    uint64_t offset = static_cast< uint64_t >(read.offset);
    return offset <= allocated->getZExtValue() &&
           size <= allocated->getZExtValue() - offset;
  }

  bool covers(const Address& store,
              uint64_t size,
              const Address& read,
              uint64_t read_size) const {
    if (store.base != read.base || store.indirect != read.indirect ||
        store.offset > read.offset) {
      return false;
    }
    // Unsigned subtraction gives the exact nonnegative mathematical
    // difference here, including when signed subtraction would overflow.
    uint64_t delta = static_cast< uint64_t >(read.offset) -
                     static_cast< uint64_t >(store.offset);
    return delta <= size && read_size <= size - delta;
  }

  bool scan(llvm::BasicBlock* block,
            llvm::BasicBlock::iterator end,
            Address read,
            uint64_t size,
            unsigned depth) {
    if (depth >= 32 || !this->consume() ||
        !this->_active.insert(block).second) {
      return false;
    }
    bool result = this->scan_block(block, end, read, size, depth);
    this->_active.erase(block);
    return result;
  }

  bool scan_block(llvm::BasicBlock* block,
                  llvm::BasicBlock::iterator end,
                  Address read,
                  uint64_t size,
                  unsigned depth) {
    if (this->excluded_null(read)) {
      return true;
    }
    for (auto it = end; it != block->begin();) {
      llvm::Instruction& inst = *--it;
      if (!this->consume()) {
        return false;
      }
      // SSA instructions inside a loop can execute more than once. An
      // earlier iteration's pointer with the same LLVM identity is not
      // necessarily the current pointer. Never cross its definition;
      // pointer PHIs are instead substituted on their incoming edges.
      if (&inst == read.base &&
          (read.indirect || !llvm::isa< llvm::PHINode >(&inst))) {
        if (this->_prove_initialized
                ? this->initialized_allocation(inst, read, size)
                : this->untouched_allocation(inst, read, size)) {
          return true;
        }
        auto call = llvm::dyn_cast< llvm::CallBase >(&inst);
        return call && this->returned_memory(call, read, size, depth);
      }
      if (auto store = llvm::dyn_cast< llvm::StoreInst >(&inst)) {
        if (!store->isSimple()) {
          return false;
        }
        // Substitute an assignment to a symbolic pointer slot. This also
        // handles q = p without confusing old and new pointees after a
        // reassignment of either slot.
        if (read.indirect && store->getPointerOperand() == read.base) {
          Address assigned;
          int64_t offset;
          if (!this->address(store->getValueOperand(), store, assigned) ||
              __builtin_add_overflow(assigned.offset, read.offset, &offset)) {
            return false;
          }
          read = {assigned.base, offset, assigned.indirect};
          if (this->excluded_null(read)) {
            return true;
          }
          continue;
        }
        // To prove uninitializedness, require the allocation to be
        // untouched. Even a partial or possibly aliasing store makes
        // that proof unknown; mixed initialization remains a warning.
        if (!this->_prove_initialized ||
            !this->initialized(store->getValueOperand())) {
          return false;
        }
        Address dest;
        if (this->address(store->getPointerOperand(), store, dest)) {
          auto bytes = this->_layout.getTypeStoreSize(
              store->getValueOperand()->getType());
          if (!bytes.isScalable() &&
              this->covers(dest, bytes.getFixedValue(), read, size)) {
            return true;
          }
        }
        // Other stores of initialized constants cannot uninitialize bytes.
      } else if (auto call = llvm::dyn_cast< llvm::CallBase >(&inst)) {
        if (!this->preserves_memory(call)) {
          return false;
        }
      } else if (inst.mayWriteToMemory()) {
        return false;
      }
    }

    if (llvm::pred_empty(block)) {
      return block == &this->_function.getEntryBlock() &&
             this->argument_memory(read, size, depth);
    }
    for (llvm::BasicBlock* pred : llvm::predecessors(block)) {
      Address incoming = read;
      if (!read.indirect) {
        if (auto phi = llvm::dyn_cast< llvm::PHINode >(read.base)) {
          if (phi->getParent() == block) {
            Address translated;
            if (!this->address(phi->getIncomingValueForBlock(pred),
                               pred->getTerminator(),
                               translated)) {
              return false;
            }
            int64_t sum;
            if (__builtin_add_overflow(translated.offset, read.offset, &sum)) {
              return false;
            }
            incoming = {translated.base, sum, translated.indirect};
          }
        }
      }
      if (!this->scan(pred, pred->end(), incoming, size, depth + 1)) {
        return false;
      }
    }
    return true;
  }

public:
  bool prove(llvm::LoadInst* load) {
    if (this->_function.hasFnAttribute(llvm::Attribute::NullPointerIsValid) ||
        !load->isSimple() || !load->getType()->isSingleValueType() ||
        load->getType()->isVectorTy()) {
      return false;
    }
    auto size = this->_layout.getTypeStoreSize(load->getType());
    Address read;
    return !size.isScalable() &&
           this->address(load->getPointerOperand(), load, read) &&
           this->scan(load->getParent(),
                      load->getIterator(),
                      read,
                      size.getFixedValue(),
                      0);
  }
};

} // namespace

bool prove_initialized_load(ar::Load* load, CallContext* context) {
  if (!load->has_frontend()) {
    return false;
  }
  auto inst = llvm::dyn_cast< llvm::LoadInst >(load->frontend< llvm::Value >());
  unsigned budget = 1024;
  return inst &&
         InitializationProof(*inst->getFunction(), context, budget, true)
             .prove(inst);
}

bool prove_uninitialized_load(ar::Load* load, CallContext* context) {
  if (!load->has_frontend()) {
    return false;
  }
  auto inst = llvm::dyn_cast< llvm::LoadInst >(load->frontend< llvm::Value >());
  unsigned budget = 1024;
  return inst &&
         InitializationProof(*inst->getFunction(), context, budget, false)
             .prove(inst);
}

} // namespace analyzer
} // namespace ikos
