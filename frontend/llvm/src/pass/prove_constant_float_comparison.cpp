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

/// \brief C++ `abs(long)`/`abs(long long)` overloads of <cstdlib>
/// (v3.5.8.4). libc++ defines them inline in the global namespace and
/// libstdc++ in std; both have the meaning of labs/llabs. An inline
/// definition from the standard header is linkonce_odr (ODR: the same
/// definition everywhere); a program cannot redefine these functions.
bool cxx_integer_abs(const Function* callee) {
  StringRef name = callee->getName();
  return (name == "_Z3absl" || name == "_ZSt3absl" || name == "_Z3absx" ||
          name == "_ZSt3absx") &&
         (callee->isDeclaration() || callee->hasLinkOnceODRLinkage() ||
          callee->hasAvailableExternallyLinkage());
}

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
    if (!callee || call->isNoBuiltin() || !call->getType()->isIntegerTy()) {
      return nullptr;
    }
    bool cxx = cxx_integer_abs(callee);
    if (!callee->isDeclaration() && !cxx) {
      return nullptr;
    }
    bool intrinsic = callee->getIntrinsicID() == Intrinsic::abs;
    StringRef name = callee->getName();
    if (!(intrinsic || cxx || name == "abs" || name == "labs" ||
          name == "llabs" || name == "imaxabs") ||
        call->arg_size() != (intrinsic ? 2 : 1) ||
        call->getArgOperand(0)->getType() != call->getType()) {
      return nullptr;
    }
    return call;
  }

  // v3.5.8.1: two simple loads of the same address read the same value if
  // nothing may write memory between them. Supported shapes: both loads in
  // one block, or the second in the only successor-reached block whose single
  // predecessor holds the first.
  bool same_loaded_value(Value* a, Value* b) {
    auto first = dyn_cast< LoadInst >(a);
    auto second = dyn_cast< LoadInst >(b);
    if (!first || !second || !first->isSimple() || !second->isSimple() ||
        first->getPointerOperand() != second->getPointerOperand() ||
        first->getType() != second->getType()) {
      return false;
    }
    if (!_dominators.dominates(first, second)) {
      std::swap(first, second);
      if (!_dominators.dominates(first, second)) {
        return false;
      }
    }
    auto clean = [](BasicBlock::iterator it, BasicBlock::iterator end) {
      for (; it != end; ++it) {
        if (it->mayWriteToMemory()) {
          return false;
        }
      }
      return true;
    };
    BasicBlock* head = first->getParent();
    BasicBlock* tail = second->getParent();
    if (head == tail) {
      return clean(std::next(first->getIterator()), second->getIterator());
    }
    return tail->getSinglePredecessor() == head &&
           clean(std::next(first->getIterator()), head->end()) &&
           clean(tail->begin(), second->getIterator());
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
      if (!comparison) {
        continue;
      }
      Value* a = this->resolve(comparison->getOperand(0));
      Value* b = this->resolve(comparison->getOperand(1));
      auto predicate = comparison->getPredicate();
      if (b == input || this->same_loaded_value(b, input)) {
        std::swap(a, b);
        predicate = CmpInst::getSwappedPredicate(predicate);
      }
      auto number = dyn_cast< ConstantInt >(b);
      if ((a != input && !this->same_loaded_value(a, input)) || !number ||
          number->getBitWidth() != width ||
          !number->getValue().isMinSignedValue()) {
        continue;
      }
      // x != MIN and x > MIN (v3.5.8.1) exclude MIN on the true edge;
      // x == MIN and x <= MIN exclude it on the false edge.
      unsigned edge;
      if (predicate == CmpInst::ICMP_NE || predicate == CmpInst::ICMP_SGT) {
        edge = 0;
      } else if (predicate == CmpInst::ICMP_EQ ||
                 predicate == CmpInst::ICMP_SLE) {
        edge = 1;
      } else {
        continue;
      }
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
    // The correctly rounded LLVM intrinsic, or (v3.5.8.3) the C library
    // sqrt/sqrtf/sqrtl declared outside the module. Clang emits the library
    // call instead of the intrinsic under -fmath-errno (the Linux default).
    // The bounds below are the two representable neighbours of the real root
    // (or integer bounds for other formats): any faithfully rounded sqrt,
    // and in particular the correctly rounded IEEE 754 one, is within them.
    // A sqrt defined in the module and constrained FP operations are unknown.
    bool library = false;
    if (callee && callee->isDeclaration() && !callee->isVarArg()) {
      auto name = callee->getName();
      auto type = call->getType();
      library = (name == "sqrt" && type->isDoubleTy()) ||
                (name == "sqrtf" && type->isFloatTy()) ||
                (name == "sqrtl" && type->isFloatingPointTy());
    }
    if (!callee ||
        (callee->getIntrinsicID() != Intrinsic::sqrt && !library) ||
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
    // v3.5.8.2: abs(MIN) is not excluded. abs(MIN) wraps to MIN, which passes
    // `abs(x) <= M`, so only `x <= M` holds on the true edge.
    bool upper_only;
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
      // v3.5.8.1: `abs(x) < (long)sqrt(...)` compares the absolute value
      // after a widening cast. abs(x) >= 0 once abs(MIN) is excluded (checked
      // below), so sign and zero extension both preserve its value. If it is
      // not excluded, sext(abs(MIN)) is negative and passes, still x <= M
      // (v3.5.8.2); zext(abs(MIN)) = 2^(w-1) exceeds any M fitting in x.
      Value* lhs = this->resolve(integer->getOperand(0));
      for (unsigned i = 0; i < 4; ++i) {
        auto ext = dyn_cast< CastInst >(lhs);
        if (!ext || (ext->getOpcode() != Instruction::SExt &&
                     ext->getOpcode() != Instruction::ZExt)) {
          break;
        }
        lhs = this->resolve(ext->getOperand(0));
      }
      absolute = this->integer_abs(lhs);
      if (!absolute) {
        return None;
      }
      Optional< APInt > limit;
      if (auto number = dyn_cast_or_null< ConstantInt >(
              this->constant(integer->getOperand(1)))) {
        limit = number->getValue();
      } else if (auto conversion = dyn_cast< FPToSIInst >(
                     this->resolve(integer->getOperand(1)))) {
        // v3.5.8.1: fptosi(sqrt(C)) <= trunc(upper bound of sqrt(C)).
        auto root = this->bounds(conversion->getOperand(0));
        if (!root || !root->upper.isFinite() || root->upper.isNegative()) {
          return None;
        }
        unsigned width = conversion->getType()->getIntegerBitWidth();
        APSInt value(width, /*isUnsigned=*/false);
        bool exact;
        auto status = root->upper.convertToInteger(value,
                                                   APFloat::rmTowardZero,
                                                   &exact);
        if (status & (APFloat::opInvalidOp | APFloat::opOverflow)) {
          return None;
        }
        limit = static_cast< APInt >(value);
      }
      if (!limit || limit->isNegative()) {
        return None;
      }
      maximum = *limit;
      if (integer->getPredicate() == CmpInst::ICMP_SLT) {
        if (maximum->isZero()) {
          return None;
        }
        --*maximum;
      }
      input = this->resolve(absolute->getArgOperand(0));
      // The constraint is applied to the abs input; fit it to that width.
      unsigned input_width = input->getType()->getIntegerBitWidth();
      if (maximum->getBitWidth() > input_width) {
        if (maximum->getActiveBits() >= input_width) {
          return None; // Not below the signed maximum: nothing to learn
        }
        maximum = maximum->trunc(input_width);
      } else if (maximum->getBitWidth() < input_width) {
        maximum = maximum->zext(input_width);
      }
    } else if (auto floating = dyn_cast< FCmpInst >(comparison)) {
      if ((floating->getPredicate() != CmpInst::FCMP_OLT &&
           floating->getPredicate() != CmpInst::FCMP_OLE) ||
          floating->getFastMathFlags().any()) {
        return None;
      }
      auto type = floating->getOperand(0)->getType();
      // v3.5.8.3: also x86 long double / quad (e.g. Linux `imaxabs(x) <=
      // sqrtl(LLONG_MAX)`). Only an exact ordered comparison of an integer
      // conversion with the bounds of the limit is used here; the bounds of
      // these formats are integer bounds (see bounds()).
      if (!(type->isHalfTy() || type->isFloatTy() || type->isDoubleTy() ||
            type->isX86_FP80Ty() || type->isFP128Ty())) {
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
    if (!input || !maximum) {
      return None;
    }
    // v3.5.8.4: `abs((long)c)` with a char/short c compares a sign extension
    // (possibly truncated back to a width >= the source), e.g.
    // `abs(trunc(sext i8 c to i64) to i32)`. Such a value is never the MIN
    // of its type, and it has the same value as c: constrain c itself, so
    // that later reads of c (not of the converted value) see the range.
    bool narrow_source = false;
    for (unsigned i = 0; i < 4; ++i) {
      Value* inner = input;
      if (auto trunc = dyn_cast< TruncInst >(inner)) {
        inner = this->resolve(trunc->getOperand(0));
        auto sext = dyn_cast< SExtInst >(inner);
        if (!sext || sext->getSrcTy()->getIntegerBitWidth() >
                         trunc->getDestTy()->getIntegerBitWidth()) {
          break;
        }
      }
      auto sext = dyn_cast< SExtInst >(inner);
      if (!sext) {
        break;
      }
      Value* source = this->resolve(sext->getOperand(0));
      unsigned source_width = source->getType()->getIntegerBitWidth();
      if (maximum->getActiveBits() >= source_width) {
        break; // Not below the signed maximum of c: nothing to learn on c
      }
      maximum = maximum->trunc(source_width);
      input = source;
      narrow_source = true;
    }
    bool upper_only = absolute && !narrow_source &&
                      !this->minimum_excluded(absolute, input);
    return Guard{branch, input, *maximum, upper_only};
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
    auto lower = ConstantInt::get(
        type,
        guard.upper_only
            ? APInt::getSignedMinValue(type->getIntegerBitWidth())
            : APInt(-guard.maximum));
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
             callee->getName() == "llabs" || callee->getName() == "imaxabs" ||
             cxx_integer_abs(callee))) {
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
