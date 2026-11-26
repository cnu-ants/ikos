/*******************************************************************************
 *
 * \file
 * \brief Division by zero checker implementation
 *
 * Author: Maxime Arthaud
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

#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>

#include <ikos/analyzer/analysis/literal.hpp>
#include <ikos/analyzer/checker/division_by_zero.hpp>
#include <ikos/analyzer/json/helper.hpp>
#include <ikos/analyzer/support/cast.hpp>
#include <ikos/analyzer/util/log.hpp>

namespace ikos {
namespace analyzer {

namespace {

enum class FloatZero { Unknown, Zero, NonZero, Unreachable };

/// A restricted proof over the original LLVM values. IKOS does not have a
/// floating point value domain. In particular, unknown inputs must never be
/// treated as nonzero just because their initializedness is known.
class FloatDivisorProof {
private:
  llvm::DominatorTree _dominators;
  const llvm::DataLayout& _layout;
  bool _ieee;
  unsigned _budget = 512;

public:
  explicit FloatDivisorProof(llvm::Function& function)
      : _dominators(function),
        _layout(function.getParent()->getDataLayout()),
        _ieee(!function.hasFnAttribute(llvm::Attribute::StrictFP)) {
    for (llvm::StringRef name : {"denormal-fp-math", "denormal-fp-math-f32"}) {
      if (function.hasFnAttribute(name)) {
        auto mode = function.getFnAttribute(name).getValueAsString();
        this->_ieee &= mode == "ieee" || mode == "ieee,ieee";
      }
    }
    if (function.hasFnAttribute("unsafe-fp-math") &&
        function.getFnAttribute("unsafe-fp-math").getValueAsString() ==
            "true") {
      this->_ieee = false;
    }
  }

private:
  // Only forward an exact, single store to a nonescaping stack slot. Every
  // access must be ordinary and the store must dominate the read. An alias,
  // volatile access, second store or escape invalidates this proof.
  llvm::Value* resolve(llvm::Value* value, unsigned depth = 0) {
    if (depth >= 32 || this->_budget == 0) {
      return value;
    }
    --this->_budget;
    auto load = llvm::dyn_cast< llvm::LoadInst >(value);
    auto slot =
        load ? llvm::dyn_cast< llvm::AllocaInst >(load->getPointerOperand())
             : nullptr;
    if (!slot || !load->isSimple() ||
        slot->getAllocatedType() != load->getType()) {
      return value;
    }
    llvm::StoreInst* definition = nullptr;
    for (llvm::User* user : slot->users()) {
      if (auto store = llvm::dyn_cast< llvm::StoreInst >(user)) {
        if (definition || !store->isSimple() ||
            store->getPointerOperand() != slot ||
            store->getValueOperand()->getType() != load->getType()) {
          return value;
        }
        definition = store;
      } else if (auto read = llvm::dyn_cast< llvm::LoadInst >(user)) {
        if (!read->isSimple() || read->getPointerOperand() != slot) {
          return value;
        }
      } else {
        return value;
      }
    }
    if (!definition || !this->_dominators.dominates(definition, load)) {
      return value;
    }
    return this->resolve(definition->getValueOperand(), depth + 1);
  }

  llvm::Constant* constant(llvm::Value* value, unsigned depth = 0) {
    if (depth >= 32 || this->_budget == 0) {
      return nullptr;
    }
    --this->_budget;
    value = this->resolve(value);
    if (llvm::isa< llvm::ConstantFP >(value) ||
        llvm::isa< llvm::ConstantInt >(value)) {
      return llvm::cast< llvm::Constant >(value);
    }
    auto instruction = llvm::dyn_cast< llvm::Instruction >(value);
    if (!instruction || !this->_ieee ||
        !(llvm::isa< llvm::CastInst >(instruction) ||
          llvm::isa< llvm::BinaryOperator >(instruction) ||
          llvm::isa< llvm::UnaryOperator >(instruction))) {
      return nullptr;
    }
    if ((llvm::isa< llvm::FPMathOperator >(instruction) &&
         instruction->getFastMathFlags().any()) ||
        instruction->getMetadata(llvm::LLVMContext::MD_fpmath)) {
      return nullptr;
    }
    llvm::SmallVector< llvm::Constant*, 2 > operands;
    for (llvm::Value* operand : instruction->operands()) {
      auto c = this->constant(operand, depth + 1);
      if (!c) {
        return nullptr;
      }
      operands.push_back(c);
    }
    // LLVM's target-format arithmetic preserves signed zero, subnormals,
    // infinities and NaNs. Do not use host double or epsilon comparisons.
    return llvm::ConstantFoldInstOperands(instruction, operands, this->_layout);
  }

  FloatZero classify_value(llvm::Value* value, unsigned depth = 0) {
    if (depth >= 32 || this->_budget == 0) {
      return FloatZero::Unknown;
    }
    --this->_budget;
    value = this->resolve(value);
    if (auto number =
            llvm::dyn_cast_or_null< llvm::ConstantFP >(this->constant(value))) {
      const auto& floating = number->getValueAPF();
      if (floating.isZero()) {
        return FloatZero::Zero;
      }
      if (!this->_ieee && floating.isDenormal()) {
        return FloatZero::Unknown;
      }
      return FloatZero::NonZero;
    }
    llvm::SmallVector< llvm::Value*, 4 > alternatives;
    if (auto select = llvm::dyn_cast< llvm::SelectInst >(value)) {
      if (auto condition = llvm::dyn_cast_or_null< llvm::ConstantInt >(
              this->constant(select->getCondition()))) {
        return this->classify_value(condition->isZero()
                                        ? select->getFalseValue()
                                        : select->getTrueValue(),
                                    depth + 1);
      }
      alternatives.push_back(select->getTrueValue());
      alternatives.push_back(select->getFalseValue());
    } else if (auto phi = llvm::dyn_cast< llvm::PHINode >(value)) {
      for (llvm::Value* incoming : phi->incoming_values()) {
        alternatives.push_back(incoming);
      }
    }
    if (alternatives.empty()) {
      return FloatZero::Unknown;
    }
    auto result = this->classify_value(alternatives.front(), depth + 1);
    for (llvm::Value* alternative : alternatives) {
      if (result == FloatZero::Unknown ||
          this->classify_value(alternative, depth + 1) != result) {
        return FloatZero::Unknown;
      }
    }
    return result;
  }

public:
  bool uses_ieee() const { return this->_ieee; }

  FloatZero classify(llvm::BinaryOperator& division) {
    llvm::Value* divisor = this->resolve(division.getOperand(1));
    auto result = this->classify_value(divisor);
    if (!this->_ieee) {
      return result;
    }
    // A dominating branch can exclude both +0 and -0 without requiring a
    // complete floating interval domain. Repeated loads are equated only by
    // resolve()'s immutable-slot proof, never by pointer spelling alone.
    for (llvm::BasicBlock& block : *division.getFunction()) {
      auto branch = llvm::dyn_cast< llvm::BranchInst >(block.getTerminator());
      auto comparison =
          branch && branch->isConditional()
              ? llvm::dyn_cast< llvm::FCmpInst >(branch->getCondition())
              : nullptr;
      if (!comparison || comparison->getFastMathFlags().any() ||
          branch->getSuccessor(0) == branch->getSuccessor(1)) {
        continue;
      }
      unsigned index;
      if (this->resolve(comparison->getOperand(0)) == divisor) {
        index = 0;
      } else if (this->resolve(comparison->getOperand(1)) == divisor) {
        index = 1;
      } else {
        continue;
      }
      auto other = llvm::dyn_cast_or_null< llvm::ConstantFP >(
          this->constant(comparison->getOperand(1 - index)));
      if (!other) {
        continue;
      }
      auto zero = llvm::ConstantFP::getZeroValueForNegation(other->getType());
      auto folded = llvm::dyn_cast_or_null< llvm::ConstantInt >(
          llvm::ConstantFoldCompareInstOperands(comparison->getPredicate(),
                                                index == 0 ? zero : other,
                                                index == 0 ? other : zero,
                                                this->_layout));
      if (!folded) {
        continue;
      }
      for (unsigned edge = 0; edge < 2; ++edge) {
        if (!this->_dominators
                 .dominates(llvm::BasicBlockEdge(&block,
                                                 branch->getSuccessor(edge)),
                            division.getParent())) {
          continue;
        }
        auto constraint = FloatZero::Unknown;
        if (folded->isZero() == (edge == 0)) {
          constraint = FloatZero::NonZero;
        } else if (other->getValueAPF().isZero() &&
                   ((comparison->getPredicate() == llvm::CmpInst::FCMP_OEQ &&
                     edge == 0) ||
                    (comparison->getPredicate() == llvm::CmpInst::FCMP_UNE &&
                     edge == 1))) {
          constraint = FloatZero::Zero;
        }
        if (constraint != FloatZero::Unknown) {
          if (result != FloatZero::Unknown && result != constraint) {
            return FloatZero::Unreachable;
          }
          result = constraint;
        }
      }
    }
    return result;
  }
};

} // namespace

DivisionByZeroChecker::DivisionByZeroChecker(Context& ctx) : Checker(ctx) {}

CheckerName DivisionByZeroChecker::name() const {
  return CheckerName::DivisionByZero;
}

const char* DivisionByZeroChecker::description() const {
  return "Division by zero checker";
}

void DivisionByZeroChecker::check(ar::Statement* stmt,
                                  const value::AbstractDomain& inv,
                                  CallContext* call_context) {
  if (auto bin = dyn_cast< ar::BinaryOperation >(stmt)) {
    if (bin->op() == ar::BinaryOperation::FDiv &&
        !bin->right()->type()->is_float()) {
      return; // The scalar proof does not model vector lanes.
    }
    if (bin->op() == ar::BinaryOperation::UDiv ||
        bin->op() == ar::BinaryOperation::SDiv ||
        bin->op() == ar::BinaryOperation::URem ||
        bin->op() == ar::BinaryOperation::SRem ||
        bin->op() == ar::BinaryOperation::FDiv) {
      CheckResult check = this->check_division(bin, inv);
      this->display_invariant(check.result, stmt, inv);
      this->_checks.insert(check.kind,
                           CheckerName::DivisionByZero,
                           check.result,
                           stmt,
                           call_context,
                           std::array< ar::Value*, 1 >{{bin->right()}},
                           check.info);
    }
  }
}

DivisionByZeroChecker::CheckResult DivisionByZeroChecker::check_division(
    ar::BinaryOperation* stmt, const value::AbstractDomain& inv) {
  if (inv.is_normal_flow_bottom()) {
    // Statement unreachable
    if (auto msg = this->display_division_check(Result::Unreachable, stmt)) {
      *msg << "\n";
    }
    return {CheckKind::Unreachable, Result::Unreachable, {}};
  }

  const ScalarLit& lit = this->_lit_factory.get_scalar(stmt->right());

  if (lit.is_undefined() ||
      (lit.is_var() && inv.normal().uninit_is_uninitialized(lit.var()))) {
    // Undefined operand
    if (auto msg = this->display_division_check(Result::Error, stmt)) {
      *msg << ": undefined operand\n";
    }
    return {CheckKind::UninitializedVariable, Result::Error, {}};
  }

  if (stmt->op() == ar::BinaryOperation::FDiv) {
    auto zero = FloatZero::Unknown;
    if (stmt->has_frontend()) {
      if (auto division = llvm::dyn_cast< llvm::BinaryOperator >(
              stmt->frontend< llvm::Value >())) {
        if (division->getOpcode() == llvm::Instruction::FDiv) {
          FloatDivisorProof proof(*division->getFunction());
          zero = proof.classify(*division);
          auto stored = lit.is_floating_point()
                            ? lit.floating_point().zero
                            : inv.normal().float_to_zero(lit.var());
          auto type = cast< ar::FloatType >(stmt->right()->type());
          auto format = static_cast< uint64_t >(type->float_semantic()) + 1;
          if (stored.is_raw_zero() || stored.format() == format) {
            auto known = FloatZero::Unknown;
            if (stored.is_zero()) {
              known = FloatZero::Zero;
            } else if (stored.is_non_zero() &&
                       (proof.uses_ieee() || !stored.may_be_subnormal())) {
              known = FloatZero::NonZero;
            }
            if (known != FloatZero::Unknown && zero != FloatZero::Unreachable) {
              zero = zero == FloatZero::Unknown || zero == known
                         ? known
                         : FloatZero::Unreachable;
            }
          }
        }
      }
    }
    if (zero == FloatZero::Unreachable) {
      if (auto msg = this->display_division_check(Result::Unreachable, stmt)) {
        *msg << ": contradictory floating point constraints\n";
      }
      return {CheckKind::Unreachable, Result::Unreachable, {}};
    }
    auto result = zero == FloatZero::Zero      ? Result::Error
                  : zero == FloatZero::NonZero ? Result::Ok
                                               : Result::Warning;
    if (auto msg = this->display_division_check(result, stmt)) {
      *msg << (zero == FloatZero::Zero ? ": floating divisor is zero\n"
               : zero == FloatZero::NonZero
                   ? ": floating divisor is nonzero\n"
                   : ": floating divisor may be zero\n");
    }
    JsonDict info;
    info.put("floating_point", true);
    return {CheckKind::DivisionByZero, result, std::move(info)};
  }

  auto divisor = IntInterval::bottom(1, Signed);
  if (lit.is_machine_int()) {
    divisor = IntInterval(lit.machine_int());
  } else if (lit.is_machine_int_var()) {
    divisor = inv.normal().int_to_interval(lit.var());
  } else {
    log::error("unexpected operand to binary operation");
    return {CheckKind::UnexpectedOperand, Result::Error, {}};
  }

  boost::optional< MachineInt > d = divisor.singleton();

  if (d && (*d).is_zero()) {
    // The second operand is definitely 0
    if (auto msg = this->display_division_check(Result::Error, stmt)) {
      *msg << ": ∀d ∈ divisor, d == 0\n";
    }
    return {CheckKind::DivisionByZero, Result::Error, {}};
  } else if (divisor.contains(
                 MachineInt::zero(divisor.bit_width(), divisor.sign()))) {
    // The second operand may be 0
    if (auto msg = this->display_division_check(Result::Warning, stmt)) {
      *msg << ": ∃d ∈ divisor, d == 0\n";
    }
    return {CheckKind::DivisionByZero, Result::Warning, to_json(divisor)};
  } else {
    // The second operand cannot be definitely 0
    if (auto msg = this->display_division_check(Result::Ok, stmt)) {
      *msg << ": ∀d ∈ divisor, d != 0\n";
    }
    return {CheckKind::DivisionByZero, Result::Ok, {}};
  }
}

llvm::Optional< LogMessage > DivisionByZeroChecker::display_division_check(
    Result result, ar::BinaryOperation* stmt) const {
  auto msg = this->display_check(result, stmt);
  if (msg) {
    *msg << "check_dbz(";
    stmt->dump(msg->stream());
    *msg << ")";
  }
  return msg;
}

} // end namespace analyzer
} // end namespace ikos
