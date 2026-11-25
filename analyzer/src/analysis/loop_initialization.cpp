#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>

#include <ikos/analyzer/analysis/initialization.hpp>
#include <ikos/ar/semantic/statement.hpp>
#include <ikos/core/value/machine_int/interval_congruence.hpp>

namespace ikos {
namespace analyzer {
namespace {

/// \brief Return true if `call` is a heap allocation with a fixed byte size
///
/// Supported: malloc(n), operator new(n), operator new[](n), with a constant
/// size. The returned object is fresh, so its contents are only reachable
/// through the returned pointer.
bool heap_allocation_size(const llvm::CallBase* call, uint64_t& size) {
  const llvm::Function* callee = call->getCalledFunction();
  // v3.5.19: trust the allocator contract (a fresh block that does not
  // overlap live memory) only for the library declaration, not for a
  // function of the same name defined in the program.
  if (!callee || !callee->isDeclaration() || call->arg_size() < 1) {
    return false;
  }
  llvm::StringRef name = callee->getName();
  if (name != "malloc" && name != "_Znwm" && name != "_Znam" &&
      name != "_Znwj" && name != "_Znaj") {
    return false;
  }
  auto bytes = llvm::dyn_cast< llvm::ConstantInt >(call->getArgOperand(0));
  if (!bytes || bytes->getBitWidth() > 64 || bytes->isZero()) {
    return false;
  }
  size = bytes->getZExtValue();
  return true;
}

/// \brief Return true if `call` only deallocates `object`
bool is_deallocation_of(const llvm::CallBase* call, const llvm::Value* object) {
  const llvm::Function* callee = call->getCalledFunction();
  if (!callee || call->arg_size() < 1 || call->getArgOperand(0) != object) {
    return false;
  }
  for (unsigned i = 1; i < call->arg_size(); ++i) {
    // Sized delete passes the size as an integer, never the object again.
    if (call->getArgOperand(i) == object) {
      return false;
    }
  }
  llvm::StringRef name = callee->getName();
  return name == "free" || name == "_ZdlPv" || name == "_ZdaPv" ||
         name == "_ZdlPvm" || name == "_ZdaPvm" || name == "_ZdlPvj" ||
         name == "_ZdaPvj";
}

/// \brief Return the root object (stack slot or fixed-size heap allocation)
llvm::Value* memory_object(llvm::Value* pointer) {
  for (unsigned depth = 0; depth < 32; ++depth) {
    if (auto gep = llvm::dyn_cast< llvm::GEPOperator >(pointer)) {
      pointer = gep->getPointerOperand();
    } else if (auto cast = llvm::dyn_cast< llvm::BitCastInst >(pointer)) {
      pointer = cast->getOperand(0);
    } else if (llvm::isa< llvm::AllocaInst >(pointer)) {
      return pointer;
    } else if (auto call = llvm::dyn_cast< llvm::CallBase >(pointer)) {
      uint64_t size;
      return heap_allocation_size(call, size) ? pointer : nullptr;
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

/// Prove coverage by completed, counted loops on a nonescaping stack object.
/// This is independent of memory-cell widening: every accepted loop starts at
/// zero, increments by one, has one header exit, and stores on every backedge.
/// Both SSA induction variables and their unoptimized stack slots are checked.
/// Unsupported shapes or exhausted budgets return unknown, never initialized.
class LoopInitializationProof {
  struct Counter {
    llvm::Loop* loop;
    llvm::Value* induction;
    llvm::AllocaInst* slot;
    llvm::StoreInst* increment;
    llvm::BasicBlock* exit;
    uint64_t count;
  };
  struct Range {
    uint64_t begin;
    uint64_t end; // exclusive
  };

  llvm::Function& _function;
  const llvm::DataLayout& _layout;
  llvm::DominatorTree _dominators;
  llvm::LoopInfo _loops;
  std::vector< Counter > _counters;
  std::vector< llvm::StoreInst* > _stores;
  unsigned _budget = 32768;

  bool consume() {
    if (!this->_budget) {
      return false;
    }
    --this->_budget;
    return true;
  }

  static llvm::Value* add_one_operand(llvm::Value* value) {
    auto add = llvm::dyn_cast< llvm::BinaryOperator >(value);
    if (!add || add->getOpcode() != llvm::Instruction::Add) {
      return nullptr;
    }
    for (unsigned i = 0; i < 2; ++i) {
      auto one = llvm::dyn_cast< llvm::ConstantInt >(add->getOperand(i));
      if (one && one->isOne()) {
        return add->getOperand(1 - i);
      }
    }
    return nullptr;
  }

  bool counter(llvm::Loop* loop, Counter& out) {
    llvm::BasicBlock* header = loop->getHeader();
    llvm::BasicBlock* preheader = loop->getLoopPreheader();
    llvm::BasicBlock* latch = loop->getLoopLatch();
    auto branch = llvm::dyn_cast< llvm::BranchInst >(header->getTerminator());
    if (!preheader || !latch || !loop->getSubLoops().empty() ||
        loop->getExitingBlock() != header || !branch ||
        !branch->isConditional() || !loop->contains(branch->getSuccessor(0)) ||
        loop->contains(branch->getSuccessor(1))) {
      return false;
    }
    auto backedge = llvm::dyn_cast< llvm::BranchInst >(latch->getTerminator());
    if (!backedge || !backedge->isUnconditional() ||
        backedge->getSuccessor(0) != header) {
      return false;
    }
    auto cmp = llvm::dyn_cast< llvm::ICmpInst >(branch->getCondition());
    if (!cmp || (cmp->getPredicate() != llvm::ICmpInst::ICMP_SLT &&
                 cmp->getPredicate() != llvm::ICmpInst::ICMP_ULT)) {
      return false;
    }
    auto bound = llvm::dyn_cast< llvm::ConstantInt >(cmp->getOperand(1));
    if (!bound || bound->getBitWidth() > 64 || bound->isZero() ||
        (cmp->isSigned() && bound->isNegative()) ||
        bound->getZExtValue() > 4096) {
      return false;
    }
    llvm::Value* induction = cmp->getOperand(0);
    out = {loop,
           induction,
           nullptr,
           nullptr,
           branch->getSuccessor(1),
           bound->getZExtValue()};
    if (auto phi = llvm::dyn_cast< llvm::PHINode >(induction)) {
      if (phi->getParent() != header || phi->getNumIncomingValues() != 2) {
        return false;
      }
      int first = phi->getBasicBlockIndex(preheader);
      int next = phi->getBasicBlockIndex(latch);
      if (first < 0 || next < 0) {
        return false;
      }
      auto zero = llvm::dyn_cast< llvm::ConstantInt >(
          phi->getIncomingValue(static_cast< unsigned >(first)));
      return zero && zero->isZero() &&
             add_one_operand(
                 phi->getIncomingValue(static_cast< unsigned >(next))) == phi;
    }
    auto load = llvm::dyn_cast< llvm::LoadInst >(induction);
    auto slot =
        load ? llvm::dyn_cast< llvm::AllocaInst >(load->getPointerOperand())
             : nullptr;
    if (!load || !load->isSimple() || load->getParent() != header || !slot ||
        !slot->isStaticAlloca() || !slot->getAllocatedType()->isIntegerTy() ||
        slot->isArrayAllocation()) {
      return false;
    }
    llvm::StoreInst* initial = nullptr;
    llvm::StoreInst* increment = nullptr;
    for (llvm::User* user : slot->users()) {
      if (!this->consume()) {
        return false;
      }
      if (auto read = llvm::dyn_cast< llvm::LoadInst >(user)) {
        if (!read->isSimple() || read->getPointerOperand() != slot) {
          return false;
        }
      } else if (auto store = llvm::dyn_cast< llvm::StoreInst >(user)) {
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
        return false;
      }
    }
    auto zero =
        initial
            ? llvm::dyn_cast< llvm::ConstantInt >(initial->getValueOperand())
            : nullptr;
    auto old = increment ? llvm::dyn_cast_or_null< llvm::LoadInst >(
                               add_one_operand(increment->getValueOperand()))
                         : nullptr;
    if (!zero || !zero->isZero() || !old || !old->isSimple() ||
        old->getPointerOperand() != slot || !loop->contains(old->getParent()) ||
        !this->_dominators.dominates(old, increment)) {
      return false;
    }
    out.slot = slot;
    out.increment = increment;
    return true;
  }

  bool is_counter(llvm::Value* value, const Counter& counter) const {
    // Widening casts preserve the nonnegative, bounded induction values.
    for (unsigned depth = 0; depth < 8; ++depth) {
      if (auto cast = llvm::dyn_cast< llvm::CastInst >(value)) {
        if (cast->getOpcode() == llvm::Instruction::SExt ||
            cast->getOpcode() == llvm::Instruction::ZExt) {
          unsigned bits = cast->getSrcTy()->getIntegerBitWidth();
          // An unsigned counter can exceed its signed maximum. Sign extension
          // would then make the GEP index negative, rather than 0..count-1.
          if (cast->getOpcode() == llvm::Instruction::SExt && bits < 64 &&
              counter.count - 1 >= (UINT64_C(1) << (bits - 1))) {
            return false;
          }
          value = cast->getOperand(0);
          continue;
        }
      }
      break;
    }
    if (!counter.slot) {
      return value == counter.induction;
    }
    auto load = llvm::dyn_cast< llvm::LoadInst >(value);
    return load && load->isSimple() &&
           load->getPointerOperand() == counter.slot &&
           counter.loop->contains(load->getParent()) &&
           this->_dominators.dominates(load, counter.increment);
  }

  bool initialized(llvm::Value* value, unsigned depth = 0) {
    if (depth >= 16 || !this->consume()) {
      return false;
    }
    if (llvm::isa< llvm::ConstantInt >(value) ||
        llvm::isa< llvm::ConstantFP >(value) ||
        llvm::isa< llvm::ConstantPointerNull >(value)) {
      return true;
    }
    for (const Counter& counter : this->_counters) {
      if (this->is_counter(value, counter)) {
        return true;
      }
    }
    auto inst = llvm::dyn_cast< llvm::Instruction >(value);
    if (!inst || !(inst->isBinaryOp() || llvm::isa< llvm::CastInst >(inst) ||
                   llvm::isa< llvm::CmpInst >(inst) ||
                   llvm::isa< llvm::SelectInst >(inst))) {
      return false;
    }
    for (llvm::Value* operand : inst->operands()) {
      if (!this->initialized(operand, depth + 1)) {
        return false;
      }
    }
    return true;
  }

  bool object_uses(llvm::Value* object) {
    llvm::SmallVector< llvm::Value*, 16 > worklist{object};
    llvm::SmallPtrSet< llvm::Value*, 32 > visited;
    while (!worklist.empty()) {
      llvm::Value* pointer = worklist.pop_back_val();
      if (!visited.insert(pointer).second) {
        continue;
      }
      for (llvm::User* user : pointer->users()) {
        if (!this->consume()) {
          return false;
        }
        if (llvm::isa< llvm::GetElementPtrInst >(user) ||
            llvm::isa< llvm::BitCastInst >(user)) {
          worklist.push_back(user);
        } else if (auto cmp = llvm::dyn_cast< llvm::ICmpInst >(user)) {
          // A null check (e.g. after malloc) neither writes nor escapes.
          if (!cmp->isEquality() ||
              !llvm::isa< llvm::ConstantPointerNull >(
                  cmp->getOperand(cmp->getOperand(0) == pointer ? 1 : 0))) {
            return false;
          }
        } else if (auto call = llvm::dyn_cast< llvm::CallBase >(user)) {
          // Deallocation ends the lifetime; it cannot initialize contents.
          // Any read after it is a use-after-free, reported by other checks.
          // Only the object itself (possibly through pointer casts) may be
          // released; a pointer into the middle of the object may not.
          if (pointer->stripPointerCasts() != object ||
              !is_deallocation_of(call, pointer)) {
            return false;
          }
        } else if (auto load = llvm::dyn_cast< llvm::LoadInst >(user)) {
          if (!load->isSimple() || load->getPointerOperand() != pointer) {
            return false;
          }
        } else if (auto store = llvm::dyn_cast< llvm::StoreInst >(user)) {
          if (!store->isSimple() || store->getPointerOperand() != pointer ||
              store->getValueOperand() == pointer) {
            return false;
          }
          this->_stores.push_back(store);
        } else {
          // Calls (including memory/lifetime intrinsics), pointer escapes,
          // pointer-to-integer conversions and unsupported aliases reject.
          return false;
        }
      }
    }
    return true;
  }

  bool footprint(llvm::Value* pointer,
                 llvm::Value* object,
                 const Counter& counter,
                 uint64_t& begin,
                 uint64_t& stride) {
    unsigned bits = this->_layout.getIndexTypeSizeInBits(pointer->getType());
    if ((bits != 32 && bits != 64) ||
        this->_layout.getPointerTypeSizeInBits(pointer->getType()) != bits) {
      return false;
    }
    int64_t constant = 0;
    stride = 0;
    for (unsigned depth = 0; depth < 16; ++depth) {
      if (!this->consume()) {
        return false;
      }
      if (pointer == object) {
        if (constant < 0) {
          return false;
        }
        begin = static_cast< uint64_t >(constant);
        return true;
      }
      if (auto cast = llvm::dyn_cast< llvm::BitCastInst >(pointer)) {
        pointer = cast->getOperand(0);
        continue;
      }
      auto gep = llvm::dyn_cast< llvm::GEPOperator >(pointer);
      if (!gep) {
        return false;
      }
      llvm::MapVector< llvm::Value*, llvm::APInt > variables;
      llvm::APInt displacement(bits, 0);
      if (!gep->collectOffset(this->_layout, bits, variables, displacement) ||
          __builtin_add_overflow(constant,
                                 displacement.getSExtValue(),
                                 &constant)) {
        return false;
      }
      for (auto& term : variables) {
        unsigned index_bits = term.first->getType()->getIntegerBitWidth();
        // GEP implicitly sign-extends indices narrower than its index type.
        if ((index_bits < bits &&
             counter.count - 1 >= (UINT64_C(1) << (index_bits - 1))) ||
            !this->is_counter(term.first, counter) ||
            term.second.isNegative() ||
            __builtin_add_overflow(stride,
                                   term.second.getZExtValue(),
                                   &stride)) {
          return false;
        }
      }
      pointer = gep->getPointerOperand();
    }
    return false;
  }

  static bool covers(const std::vector< Range >& ranges,
                     uint64_t begin,
                     uint64_t end) {
    auto after = std::upper_bound(ranges.begin(),
                                  ranges.end(),
                                  begin,
                                  [](uint64_t b, const Range& r) {
                                    return b < r.begin;
                                  });
    return after != ranges.begin() && end <= std::prev(after)->end;
  }

public:
  explicit LoopInitializationProof(llvm::Function& function)
      : _function(function),
        _layout(function.getParent()->getDataLayout()),
        _dominators(function),
        _loops(_dominators) {}

  bool prove(llvm::LoadInst* load,
             const core::machine_int::IntervalCongruence& offset) {
    if (!load->isSimple() || !load->getType()->isSingleValueType() ||
        load->getType()->isVectorTy() || offset.is_bottom() ||
        offset.sign() != core::Unsigned ||
        this->_function.hasFnAttribute(llvm::Attribute::NullPointerIsValid)) {
      return false;
    }
    llvm::Value* object = memory_object(load->getPointerOperand());
    auto object_inst = llvm::dyn_cast_or_null< llvm::Instruction >(object);
    if (!object_inst || object_inst->getFunction() != &this->_function ||
        // An allocation inside a loop creates a new object per iteration.
        this->_loops.getLoopFor(object_inst->getParent()) != nullptr) {
      return false;
    }
    uint64_t allocation = 0;
    if (auto slot = llvm::dyn_cast< llvm::AllocaInst >(object)) {
      if (slot->getAddressSpace() != 0) {
        return false;
      }
      auto element = this->_layout.getTypeAllocSize(slot->getAllocatedType());
      if (element.isScalable()) {
        return false;
      }
      allocation = element.getFixedValue();
      if (slot->isArrayAllocation()) {
        // alloca(n) with a constant n, e.g. ALLOCA(10 * sizeof(int)).
        auto count = llvm::dyn_cast< llvm::ConstantInt >(slot->getArraySize());
        if (!count || count->getBitWidth() > 64 ||
            __builtin_mul_overflow(allocation,
                                   count->getZExtValue(),
                                   &allocation)) {
          return false;
        }
      }
    } else if (!heap_allocation_size(llvm::cast< llvm::CallBase >(object),
                                     allocation)) {
      return false;
    }
    auto bytes = this->_layout.getTypeStoreSize(load->getType());
    if (bytes.isScalable() || allocation == 0 || allocation > 1024 * 1024) {
      return false;
    }
    for (llvm::BasicBlock& block : this->_function) {
      for (llvm::Instruction& inst : block) {
        if (!this->consume()) {
          return false;
        }
        if (auto call = llvm::dyn_cast< llvm::CallBase >(&inst)) {
          if (call->isInlineAsm() ||
              call->hasFnAttr(llvm::Attribute::ReturnsTwice)) {
            return false;
          }
        }
        if (auto intrinsic = llvm::dyn_cast< llvm::IntrinsicInst >(&inst)) {
          if (intrinsic->getIntrinsicID() == llvm::Intrinsic::stackrestore ||
              intrinsic->getIntrinsicID() == llvm::Intrinsic::stacksave) {
            return false;
          }
        }
      }
    }
    llvm::SmallVector< llvm::Loop*, 8 > pending;
    for (llvm::Loop* loop : this->_loops) {
      pending.push_back(loop);
    }
    while (!pending.empty()) {
      llvm::Loop* loop = pending.pop_back_val();
      if (!this->consume()) {
        return false;
      }
      Counter induction;
      if (this->counter(loop, induction)) {
        this->_counters.push_back(induction);
      }
      for (llvm::Loop* child : loop->getSubLoops()) {
        pending.push_back(child);
      }
    }
    if (this->_counters.empty() || !this->object_uses(object)) {
      return false;
    }
    std::vector< Range > coverage;
    for (llvm::StoreInst* store : this->_stores) {
      // No other write may introduce an uninitialized value, even after an
      // initializing loop. The object cannot be modified through a call.
      if (!this->initialized(store->getValueOperand())) {
        return false;
      }
      for (const Counter& counter : this->_counters) {
        if (!counter.loop->contains(store->getParent()) ||
            counter.loop->contains(load->getParent()) ||
            !this->_dominators.dominates(counter.loop->getHeader(),
                                         load->getParent()) ||
            !this->_dominators.dominates(counter.exit, load->getParent()) ||
            !this->_dominators.dominates(store->getParent(),
                                         counter.loop->getLoopLatch()) ||
            (counter.increment &&
             !this->_dominators.dominates(store, counter.increment))) {
          continue;
        }
        uint64_t begin, stride;
        if (!this->footprint(store->getPointerOperand(),
                             object,
                             counter,
                             begin,
                             stride)) {
          continue;
        }
        auto stored =
            this->_layout.getTypeStoreSize(store->getValueOperand()->getType());
        if (stored.isScalable()) {
          return false;
        }
        for (uint64_t i = 0; i < counter.count; ++i) {
          uint64_t position, end;
          if (!this->consume() ||
              __builtin_mul_overflow(i, stride, &position) ||
              __builtin_add_overflow(position, begin, &position) ||
              __builtin_add_overflow(position, stored.getFixedValue(), &end) ||
              end > allocation) {
            return false;
          }
          coverage.push_back({position, end});
        }
      }
    }
    std::sort(coverage.begin(),
              coverage.end(),
              [](const Range& a, const Range& b) { return a.begin < b.begin; });
    std::vector< Range > merged;
    for (const Range& range : coverage) {
      if (!merged.empty() && range.begin <= merged.back().end) {
        merged.back().end = std::max(merged.back().end, range.end);
      } else {
        merged.push_back(range);
      }
    }
    if (merged.empty()) {
      return false;
    }
    auto lower = offset.interval().lb().to_z_number();
    auto upper = offset.interval().ub().to_z_number();
    auto end = upper + bytes.getFixedValue();
    if (end > allocation || !lower.fits< uint64_t >()) {
      return false;
    }
    uint64_t first = lower.to< uint64_t >();
    uint64_t last = end.to< uint64_t >();
    if (covers(merged, first, last)) {
      return true;
    }
    auto step = offset.modulus();
    if (step == 0 || (upper - lower) / step >= 4096) {
      return false;
    }
    for (auto current = lower; current <= upper; current += step) {
      if (!this->consume() ||
          !covers(merged,
                  current.to< uint64_t >(),
                  current.to< uint64_t >() + bytes.getFixedValue())) {
        return false;
      }
    }
    return true;
  }
};

} // namespace

bool prove_loop_initialized_load(
    ar::Load* load, const core::machine_int::IntervalCongruence& offset) {
  if (!load->has_frontend()) {
    return false;
  }
  auto inst = llvm::dyn_cast< llvm::LoadInst >(load->frontend< llvm::Value >());
  return inst && memory_object(inst->getPointerOperand()) &&
         LoopInitializationProof(*inst->getFunction()).prove(inst, offset);
}

} // namespace analyzer
} // namespace ikos
