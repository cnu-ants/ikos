/*******************************************************************************
 *
 * \file
 * \brief Conservative floating comparisons and integer math guard constraints
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

#include <cstdint>
#include <utility>

#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>

#include <ikos/frontend/llvm/pass.hpp>

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace llvm;

namespace {

struct FloatBounds {
  APFloat lower;
  APFloat upper;
};

/// Restricted constant proofs and necessary integer ranges for math guards.
/// Unsupported operations, unresolved aliases and special floating values are
/// left to the original analysis. No host floating arithmetic is used.
class ConstantFloatProof {
private:
  const DataLayout& _layout;
  DominatorTree _dominators;

public:
  explicit ConstantFloatProof(Function& f)
      : _layout(f.getParent()->getDataLayout()), _dominators(f) {}

private:
  // Forward only an unaliased, single definition that dominates this read.
  // This supports --opt=none without assuming repeated arbitrary loads agree.
  Value* resolve(Value* value, unsigned depth = 0) {
    if (depth > 32) {
      return value;
    }
    auto load = dyn_cast< LoadInst >(value);
    auto slot =
        load ? dyn_cast< AllocaInst >(load->getPointerOperand()) : nullptr;
    if (!slot || !load->isSimple() ||
        slot->getAllocatedType() != load->getType()) {
      return value;
    }
    StoreInst* definition = nullptr;
    for (User* user : slot->users()) {
      if (auto store = dyn_cast< StoreInst >(user)) {
        if (definition || !store->isSimple() ||
            store->getPointerOperand() != slot) {
          return value;
        }
        definition = store;
      } else if (auto read = dyn_cast< LoadInst >(user)) {
        if (!read->isSimple() || read->getPointerOperand() != slot) {
          return value;
        }
      } else {
        return value;
      }
    }
    if (!definition || !_dominators.dominates(definition, load)) {
      return value;
    }
    return this->resolve(definition->getValueOperand(), depth + 1);
  }

  CallBase* integer_abs(Value* value) {
    auto call = dyn_cast< CallBase >(this->resolve(value));
    auto callee = call ? call->getCalledFunction() : nullptr;
    if (!callee || !callee->isDeclaration() || call->isNoBuiltin() ||
        !call->getType()->isIntegerTy()) {
      return nullptr;
    }
    bool intrinsic = callee->getIntrinsicID() == Intrinsic::abs;
    StringRef name = callee->getName();
    if (!(intrinsic || name == "abs" || name == "labs" || name == "llabs" ||
          name == "imaxabs") ||
        call->arg_size() != (intrinsic ? 2 : 1) ||
        call->getArgOperand(0)->getType() != call->getType()) {
      return nullptr;
    }
    return call;
  }

  // The new constraints must not hide the unrepresentable abs(MIN) case.
  bool minimum_excluded(CallBase* absolute, Value* input) {
    unsigned width = input->getType()->getIntegerBitWidth();
    if (auto number = dyn_cast< ConstantInt >(input)) {
      return !number->getValue().isMinSignedValue();
    }
    for (BasicBlock& block : *absolute->getFunction()) {
      auto branch = dyn_cast< BranchInst >(block.getTerminator());
      auto comparison = branch && branch->isConditional()
                            ? dyn_cast< ICmpInst >(branch->getCondition())
                            : nullptr;
      if (!comparison || !comparison->isEquality()) {
        continue;
      }
      Value* a = this->resolve(comparison->getOperand(0));
      Value* b = this->resolve(comparison->getOperand(1));
      if (b == input) {
        std::swap(a, b);
      }
      auto number = dyn_cast< ConstantInt >(b);
      if (a != input || !number || number->getBitWidth() != width ||
          !number->getValue().isMinSignedValue()) {
        continue;
      }
      unsigned edge = comparison->getPredicate() == CmpInst::ICMP_NE ? 0 : 1;
      if (_dominators.dominates(BasicBlockEdge(&block,
                                               branch->getSuccessor(edge)),
                                absolute->getParent())) {
        return true;
      }
    }
    return false;
  }

  Constant* constant(Value* value, unsigned depth = 0) {
    if (depth > 32) {
      return nullptr;
    }
    value = this->resolve(value);
    if (isa< ConstantInt >(value) || isa< ConstantFP >(value)) {
      return cast< Constant >(value);
    }
    if (auto conversion = dyn_cast< CastInst >(value)) {
      if (!(conversion->getSrcTy()->isIntegerTy() ||
            conversion->getSrcTy()->isFloatingPointTy()) ||
          !(conversion->getDestTy()->isIntegerTy() ||
            conversion->getDestTy()->isFloatingPointTy())) {
        return nullptr;
      }
      auto operand = this->constant(conversion->getOperand(0), depth + 1);
      if (!operand) {
        return nullptr;
      }
      // LLVM applies the target FP semantics and the rounding of ordinary IR
      // casts. Poison/undef and non-numeric constant expressions are rejected.
      auto folded = ConstantFoldInstOperands(conversion, {operand}, _layout);
      return folded && (isa< ConstantInt >(folded) || isa< ConstantFP >(folded))
                 ? folded
                 : nullptr;
    }
    auto call = this->integer_abs(value);
    if (!call) {
      return nullptr;
    }
    auto input = dyn_cast_or_null< ConstantInt >(
        this->constant(call->getArgOperand(0), depth + 1));
    if (!input || input->getValue().isMinSignedValue()) {
      // Signed abs(MIN) is not representable. Never infer a wrapped absolute
      // value or remove its surrounding diagnostics.
      return nullptr;
    }
    return ConstantInt::get(call->getType(), input->getValue().abs());
  }

  static uint64_t floor_sqrt(uint64_t n) {
    uint64_t lower = 0;
    uint64_t upper = uint64_t(1) << 32;
    while (lower < upper) {
      uint64_t middle = lower + (upper - lower + 1) / 2;
      if (middle > n / middle) {
        upper = middle - 1;
      } else {
        lower = middle;
      }
    }
    return lower;
  }

  Optional< FloatBounds > bounds(Value* value) {
    value = this->resolve(value);
    if (auto number = dyn_cast_or_null< ConstantFP >(this->constant(value))) {
      const APFloat& x = number->getValueAPF();
      if (x.isFinite()) {
        return FloatBounds{x, x};
      }
      return None;
    }
    auto call = dyn_cast< CallBase >(value);
    auto callee = call ? call->getCalledFunction() : nullptr;
    // Only the correctly rounded LLVM intrinsic is supported here. External
    // sqrt/sqrtl implementations and constrained FP operations remain unknown.
    if (!callee || callee->getIntrinsicID() != Intrinsic::sqrt ||
        call->arg_size() != 1 || call->getFastMathFlags().any() ||
        call->getMetadata(LLVMContext::MD_fpmath) ||
        call->getArgOperand(0)->getType() != call->getType()) {
      return None;
    }
    auto input =
        dyn_cast_or_null< ConstantFP >(this->constant(call->getArgOperand(0)));
    if (!input) {
      return None;
    }
    const APFloat& x = input->getValueAPF();
    if (!x.isFinite() || (x.isNegative() && !x.isZero())) {
      return None;
    }
    APSInt floor_input(64, true);
    APSInt ceil_input(64, true);
    bool exact;
    auto floor_status =
        x.convertToInteger(floor_input, APFloat::rmTowardNegative, &exact);
    auto ceil_status =
        x.convertToInteger(ceil_input, APFloat::rmTowardPositive, &exact);
    if ((floor_status | ceil_status) &
        (APFloat::opInvalidOp | APFloat::opOverflow)) {
      return None;
    }
    uint64_t lo = floor_sqrt(floor_input.getZExtValue());
    uint64_t hi_input = ceil_input.getZExtValue();
    uint64_t hi = floor_sqrt(hi_input);
    if (hi * hi != hi_input) {
      ++hi;
    }
    APFloat lower(x.getSemantics());
    APFloat upper(x.getSemantics());
    lower.convertFromAPInt(APInt(64, lo), false, APFloat::rmTowardNegative);
    upper.convertFromAPInt(APInt(64, hi), false, APFloat::rmTowardPositive);
    // Refine half/float/double to adjacent representable bounds. A square of
    // one of these formats is exact in IEEE quad (at most 106 significant
    // bits). This is target arithmetic; no host sqrt or epsilon is used.
    unsigned width = value->getType()->getScalarSizeInBits();
    if (width == 16 || width == 32 || width == 64) {
      uint64_t low_bits = lower.bitcastToAPInt().getZExtValue();
      uint64_t high_bits = upper.bitcastToAPInt().getZExtValue();
      APFloat exact_input = x;
      bool loses_info;
      exact_input.convert(APFloat::IEEEquad(),
                          APFloat::rmNearestTiesToEven,
                          &loses_info);
      while (low_bits + 1 < high_bits) {
        uint64_t middle = low_bits + (high_bits - low_bits) / 2;
        APFloat candidate(x.getSemantics(), APInt(width, middle));
        candidate.convert(APFloat::IEEEquad(),
                          APFloat::rmNearestTiesToEven,
                          &loses_info);
        candidate.multiply(candidate, APFloat::rmNearestTiesToEven);
        auto comparison = candidate.compare(exact_input);
        if (comparison == APFloat::cmpEqual) {
          low_bits = high_bits = middle;
        } else if (comparison == APFloat::cmpLessThan) {
          low_bits = middle;
        } else {
          high_bits = middle;
        }
      }
      lower = APFloat(x.getSemantics(), APInt(width, low_bits));
      upper = APFloat(x.getSemantics(), APInt(width, high_bits));
    }
    // Correct rounding cannot leave the representable bounds of the real root.
    return FloatBounds{std::move(lower), std::move(upper)};
  }

public:
  struct Guard {
    BranchInst* branch;
    Value* input;
    APInt maximum;
  };

  Optional< Guard > guard(BranchInst* branch) {
    if (!branch->isConditional() ||
        branch->getSuccessor(0) == branch->getSuccessor(1)) {
      return None;
    }
    auto comparison = dyn_cast< CmpInst >(branch->getCondition());
    if (!comparison) {
      return None;
    }
    CallBase* absolute = nullptr;
    Value* input = nullptr;
    Optional< APInt > maximum;
    if (auto integer = dyn_cast< ICmpInst >(comparison)) {
      if (integer->getPredicate() != CmpInst::ICMP_SLT &&
          integer->getPredicate() != CmpInst::ICMP_SLE) {
        return None;
      }
      absolute = this->integer_abs(integer->getOperand(0));
      auto limit = dyn_cast_or_null< ConstantInt >(
          this->constant(integer->getOperand(1)));
      if (!absolute || !limit || limit->getValue().isNegative()) {
        return None;
      }
      maximum = limit->getValue();
      if (integer->getPredicate() == CmpInst::ICMP_SLT) {
        if (maximum->isZero()) {
          return None;
        }
        --*maximum;
      }
      input = this->resolve(absolute->getArgOperand(0));
    } else if (auto floating = dyn_cast< FCmpInst >(comparison)) {
      if ((floating->getPredicate() != CmpInst::FCMP_OLT &&
           floating->getPredicate() != CmpInst::FCMP_OLE) ||
          floating->getFastMathFlags().any()) {
        return None;
      }
      auto type = floating->getOperand(0)->getType();
      if (!(type->isHalfTy() || type->isFloatTy() || type->isDoubleTy())) {
        return None;
      }
      Value* lhs = this->resolve(floating->getOperand(0));
      bool float_absolute = false;
      if (auto call = dyn_cast< CallBase >(lhs)) {
        auto callee = call->getCalledFunction();
        if (callee && callee->getIntrinsicID() == Intrinsic::fabs &&
            !call->getFastMathFlags().any() &&
            !call->getMetadata(LLVMContext::MD_fpmath)) {
          float_absolute = true;
          lhs = this->resolve(call->getArgOperand(0));
        }
      }
      auto conversion = dyn_cast< SIToFPInst >(lhs);
      if (!conversion) {
        return None;
      }
      Value* integer = this->resolve(conversion->getOperand(0));
      if (float_absolute) {
        input = integer;
      } else {
        absolute = this->integer_abs(integer);
        if (!absolute) {
          return None;
        }
        input = this->resolve(absolute->getArgOperand(0));
      }
      unsigned width = input->getType()->getIntegerBitWidth();
      auto limit = this->bounds(floating->getOperand(1));
      if (width < 2 || width > 64 || !limit || !limit->upper.isFinite() ||
          limit->upper.isNegative()) {
        return None;
      }
      auto allowed = [&](uint64_t magnitude) {
        APFloat rounded(limit->upper.getSemantics());
        rounded.convertFromAPInt(APInt(64, magnitude),
                                 false,
                                 APFloat::rmNearestTiesToEven);
        return rounded.compare(limit->upper) != APFloat::cmpGreaterThan;
      };
      // The signed minimum has one more unit of magnitude than MAX. If it
      // passes, there is no representable symmetric signed constraint here.
      if (allowed(uint64_t(1) << (width - 1))) {
        return None;
      }
      uint64_t lower = 0;
      uint64_t upper = APInt::getSignedMaxValue(width).getZExtValue();
      while (lower < upper) {
        uint64_t middle = lower + (upper - lower + 1) / 2;
        if (allowed(middle)) {
          lower = middle;
        } else {
          upper = middle - 1;
        }
      }
      maximum = APInt(width, lower);
    }
    if (!input || !maximum ||
        (absolute && !this->minimum_excluded(absolute, input))) {
      return None;
    }
    return Guard{branch, input, *maximum};
  }

  // The original condition is retained. Add necessary integer constraints on
  // its true edge only; no unsupported call, comparison, or error is removed.
  static void strengthen(const Guard& guard) {
    auto origin = guard.branch->getParent();
    auto function = origin->getParent();
    auto yes = guard.branch->getSuccessor(0);
    auto no = guard.branch->getSuccessor(1);
    auto high =
        BasicBlock::Create(function->getContext(), "math.upper", function, yes);
    auto low =
        BasicBlock::Create(function->getContext(), "math.lower", function, yes);
    IRBuilder<> builder(high);
    builder.SetCurrentDebugLocation(guard.branch->getDebugLoc());
    auto type = guard.input->getType();
    auto upper = ConstantInt::get(type, guard.maximum);
    auto lower = ConstantInt::get(type, -guard.maximum);
    auto upper_check = builder.CreateICmpSLE(guard.input, upper);
    builder.CreateCondBr(upper_check, low, no);
    builder.SetInsertPoint(low);
    auto lower_check = builder.CreateICmpSGE(guard.input, lower);
    builder.CreateCondBr(lower_check, yes, no);
    for (PHINode& phi : yes->phis()) {
      phi.setIncomingBlock(phi.getBasicBlockIndex(origin), low);
    }
    for (PHINode& phi : no->phis()) {
      Value* value = phi.getIncomingValueForBlock(origin);
      phi.addIncoming(value, high);
      phi.addIncoming(value, low);
    }
    guard.branch->setSuccessor(0, high);
  }

  Optional< bool > prove(FCmpInst& comparison) {
    auto type = comparison.getOperand(0)->getType();
    // Use formats with the standard IEEE-754 guarantees. Extended/non-IEEE
    // formats remain outside this restricted proof.
    if (!(type->isHalfTy() || type->isFloatTy() || type->isDoubleTy() ||
          type->isFP128Ty()) ||
        comparison.getFastMathFlags().any()) {
      return None;
    }
    auto left = this->bounds(comparison.getOperand(0));
    auto right = this->bounds(comparison.getOperand(1));
    if (!left || !right) {
      return None;
    }
    auto predicate = comparison.getPredicate();
    bool disjoint = left->upper.compare(right->lower) == APFloat::cmpLessThan ||
                    right->upper.compare(left->lower) == APFloat::cmpLessThan;
    bool equal_points =
        left->lower.compare(left->upper) == APFloat::cmpEqual &&
        right->lower.compare(right->upper) == APFloat::cmpEqual &&
        left->lower.compare(right->lower) == APFloat::cmpEqual;
    if (predicate == CmpInst::FCMP_OEQ || predicate == CmpInst::FCMP_UEQ) {
      return equal_points ? Optional< bool >(true)
             : disjoint   ? Optional< bool >(false)
                          : None;
    }
    if (predicate == CmpInst::FCMP_ONE || predicate == CmpInst::FCMP_UNE) {
      return disjoint       ? Optional< bool >(true)
             : equal_points ? Optional< bool >(false)
                            : None;
    }
    // All values are finite. For the remaining order predicates, agreement at
    // every interval corner proves agreement throughout the intervals.
    Optional< bool > result;
    for (const APFloat* a : {&left->lower, &left->upper}) {
      for (const APFloat* b : {&right->lower, &right->upper}) {
        auto first = ConstantFP::get(comparison.getContext(), *a);
        auto second = ConstantFP::get(comparison.getContext(), *b);
        auto folded = dyn_cast< ConstantInt >(
            ConstantFoldCompareInstOperands(predicate, first, second, _layout));
        if (!folded || (result && *result != !folded->isZero())) {
          return None;
        }
        result = !folded->isZero();
      }
    }
    return result;
  }
};

struct ProveConstantFloatComparisonPass final : public FunctionPass {
  static char ID;
  ProveConstantFloatComparisonPass() : FunctionPass(ID) {}

  bool runOnFunction(Function& function) override {
    if (function.isDeclaration() ||
        function.hasFnAttribute(Attribute::StrictFP)) {
      return false;
    }
    // Do not reinterpret a non-default FP environment as ordinary IEEE
    // arithmetic (for example, denormals treated as zero by the target).
    for (StringRef name : {"denormal-fp-math", "denormal-fp-math-f32"}) {
      if (function.hasFnAttribute(name)) {
        auto mode = function.getFnAttribute(name).getValueAsString();
        if (mode != "ieee" && mode != "ieee,ieee") {
          return false;
        }
      }
    }
    if (function.hasFnAttribute("unsafe-fp-math") &&
        function.getFnAttribute("unsafe-fp-math").getValueAsString() ==
            "true") {
      return false;
    }
    SmallVector< FCmpInst*, 8 > comparisons;
    for (Instruction& instruction : instructions(function)) {
      if (auto comparison = dyn_cast< FCmpInst >(&instruction)) {
        if (!comparison->use_empty()) {
          comparisons.push_back(comparison);
        }
      }
    }
    bool has_integer_abs = false;
    for (Instruction& instruction : instructions(function)) {
      if (auto call = dyn_cast< CallBase >(&instruction)) {
        auto callee = call->getCalledFunction();
        if (callee &&
            (callee->getIntrinsicID() == Intrinsic::abs ||
             callee->getName() == "abs" || callee->getName() == "labs" ||
             callee->getName() == "llabs" || callee->getName() == "imaxabs")) {
          has_integer_abs = true;
          break;
        }
      }
    }
    if (comparisons.empty() && !has_integer_abs) {
      return false;
    }
    ConstantFloatProof proof(function);
    bool changed = false;
    for (auto comparison : comparisons) {
      if (auto result = proof.prove(*comparison)) {
        comparison->replaceAllUsesWith(
            ConstantInt::get(comparison->getType(), *result));
        changed = true;
      }
    }
    SmallVector< ConstantFloatProof::Guard, 8 > guards;
    for (BasicBlock& block : function) {
      if (auto branch = dyn_cast< BranchInst >(block.getTerminator())) {
        if (auto guard = proof.guard(branch)) {
          guards.push_back(std::move(*guard));
        }
      }
    }
    // Collect before editing the CFG, so all dominance queries see the same
    // unchanged function. No analysis computed above is used after this point.
    for (const auto& guard : guards) {
      ConstantFloatProof::strengthen(guard);
      changed = true;
    }
    return changed;
  }
};

} // end anonymous namespace

char ProveConstantFloatComparisonPass::ID = 0;

INITIALIZE_PASS(
    ProveConstantFloatComparisonPass,
    "prove-constant-float-comparison",
    "Prove floating comparisons and strengthen supported math guards",
    false,
    false)

FunctionPass* ikos::frontend::pass::
    create_prove_constant_float_comparison_pass() {
  return new ProveConstantFloatComparisonPass();
}
