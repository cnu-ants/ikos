/*******************************************************************************
 *
 * \file
 * \brief Generic implementation of non-relational domains for machine integers
 *
 * Author: Maxime Arthaud
 *
 * Contact: ikos@lists.nasa.gov
 *
 * Notices:
 *
 * Copyright (c) 2018-2019 United States Government as represented by the
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

#pragma once

#include <map>
#include <tuple>

#include <ikos/core/adt/patricia_tree/map.hpp>
#include <ikos/core/adt/patricia_tree/set.hpp>
#include <ikos/core/domain/abstract_domain.hpp>
#include <ikos/core/domain/machine_int/operator.hpp>
#include <ikos/core/linear_expression.hpp>
#include <ikos/core/number/machine_int.hpp>
#include <ikos/core/semantic/machine_int/variable.hpp>
#include <ikos/core/semantic/variable.hpp>

namespace ikos {
namespace core {
namespace machine_int {

/// \brief Separate machine integer values augmented with copy equalities
template < typename VariableRef, typename Value >
class SeparateDomain final
    : public core::AbstractDomain< SeparateDomain< VariableRef, Value > > {
public:
  static_assert(
      core::IsVariable< VariableRef >::value,
      "VariableRef does not meet the requirements for variable types");
  static_assert(machine_int::IsVariable< VariableRef >::value,
                "VariableRef must implement machine_int::VariableTraits");
  static_assert(core::IsAbstractDomain< Value >::value,
                "Value must implement AbstractDomain");

private:
  using VariableTrait = machine_int::VariableTraits< VariableRef >;
  using PatriciaTreeMapT = PatriciaTreeMap< VariableRef, Value >;
  using CopySet = PatriciaTreeSet< VariableRef >;
  using CopyMap = PatriciaTreeMap< VariableRef, CopySet >;
  using ClassKey =
      std::tuple< uint64_t, Signedness, boost::optional< ZNumber >, Index >;

public:
  using LinearExpressionT = LinearExpression< MachineInt, VariableRef >;
  using Iterator = typename PatriciaTreeMapT::Iterator;

private:
  PatriciaTreeMapT _tree;
  bool _is_bottom;

  // Copies have equal values until a member is overwritten or forgotten.
  // Singleton values already imply equality, so their classes are omitted.
  CopyMap _copies;

  ClassKey class_key(VariableRef x) const {
    auto n = this->get(x).singleton();
    if (n) {
      return {VariableTrait::bit_width(x),
              VariableTrait::sign(x),
              n->to_z_number(),
              0};
    }
    auto members = this->_copies.at(x);
    return {VariableTrait::bit_width(x),
            VariableTrait::sign(x),
            boost::none,
            IndexableTraits< VariableRef >::index(members ? *members->begin()
                                                          : x)};
  }

  void forget_copy(VariableRef x) {
    auto members = this->_copies.at(x);
    if (!members) {
      return;
    }
    CopySet remaining = *members;
    remaining.erase(x);
    this->_copies.erase(x);
    for (VariableRef member : remaining) {
      if (remaining.size() > 1) {
        this->_copies.insert_or_assign(member, remaining);
      } else {
        this->_copies.erase(member);
      }
    }
  }

  // Change a value without invalidating its copy relation.
  void set_value(VariableRef x, const Value& value) {
    if (value.is_bottom()) {
      this->set_to_bottom();
    } else if (value.is_top()) {
      this->_tree.erase(x);
    } else {
      this->_tree.insert_or_assign(x, value);
    }
  }

  void merge_copies(CopySet members) {
    CopySet all = members;
    for (VariableRef member : members) {
      if (auto group = this->_copies.at(member)) {
        all.join_with(*group);
      }
    }
    Value value = this->get(*all.begin());
    for (VariableRef member : all) {
      value.meet_with(this->get(member));
    }
    if (value.is_bottom()) {
      this->set_to_bottom();
      return;
    }
    for (VariableRef member : all) {
      this->set_value(member, value);
      this->_copies.erase(member);
    }
    if (!value.singleton() && all.size() > 1) {
      for (VariableRef member : all) {
        this->_copies.insert_or_assign(member, all);
      }
    }
  }

  void reduce_copies() {
    CopyMap copies = this->_copies;
    for (const auto& entry : copies) {
      if (entry.first == *entry.second.begin()) {
        this->merge_copies(entry.second);
        if (this->is_bottom()) {
          return;
        }
      }
    }
  }

  // Keep equalities valid on both paths. Equal singleton values also imply
  // equalities, even though no explicit copy class is stored for them.
  void intersect_copies(const SeparateDomain& other) {
    CopySet variables;
    for (const auto& entry : this->_tree) {
      variables.insert(entry.first);
    }
    for (const auto& entry : other._tree) {
      variables.insert(entry.first);
    }
    for (const auto& entry : this->_copies) {
      variables.insert(entry.first);
    }
    for (const auto& entry : other._copies) {
      variables.insert(entry.first);
    }
    std::map< std::pair< ClassKey, ClassKey >, CopySet > groups;
    for (VariableRef variable : variables) {
      Value value = this->get(variable);
      if (value.singleton() && value.equals(other.get(variable))) {
        continue; // The merged value stays a singleton, so no class is needed.
      }
      groups[{this->class_key(variable), other.class_key(variable)}].insert(
          variable);
    }
    CopyMap copies;
    for (const auto& entry : groups) {
      if (entry.second.size() > 1) {
        for (VariableRef member : entry.second) {
          copies.insert_or_assign(member, entry.second);
        }
      }
    }
    this->_copies = std::move(copies);
  }

  void meet_copies(const SeparateDomain& other) {
    CopyMap copies = other._copies;
    this->reduce_copies();
    for (const auto& entry : copies) {
      if (this->is_bottom()) {
        return;
      }
      if (entry.first == *entry.second.begin()) {
        this->merge_copies(entry.second);
      }
    }
  }

private:
  struct TopTag {};
  struct BottomTag {};
  struct BottomFound {};

  /// \brief Create the top abstract value
  explicit SeparateDomain(TopTag) : _is_bottom(false) {}

  /// \brief Create the bottom abstract value
  explicit SeparateDomain(BottomTag) : _is_bottom(true) {}

public:
  /// \brief Create the top abstract value
  static SeparateDomain top() { return SeparateDomain(TopTag{}); }

  /// \brief Create the bottom abstract value
  static SeparateDomain bottom() { return SeparateDomain(BottomTag{}); }

  /// \brief Copy constructor
  SeparateDomain(const SeparateDomain&) noexcept = default;

  /// \brief Move constructor
  SeparateDomain(SeparateDomain&&) noexcept = default;

  /// \brief Copy assignment operator
  SeparateDomain& operator=(const SeparateDomain&) noexcept = default;

  /// \brief Move assignment operator
  SeparateDomain& operator=(SeparateDomain&&) noexcept = default;

  /// \brief Destructor
  ~SeparateDomain() override = default;

  /// \brief Begin iterator over the pairs (variable, value)
  Iterator begin() const {
    ikos_assert(!this->is_bottom());
    return this->_tree.begin();
  }

  /// \brief End iterator over the pairs (variable, value)
  Iterator end() const {
    ikos_assert(!this->is_bottom());
    return this->_tree.end();
  }

  void normalize() override {}

  bool is_bottom() const override { return this->_is_bottom; }

  bool is_top() const override {
    return !this->is_bottom() && this->_tree.empty() && this->_copies.empty();
  }

  void set_to_bottom() override {
    this->_is_bottom = true;
    this->_tree.clear();
    this->_copies.clear();
  }

  void set_to_top() override {
    this->_is_bottom = false;
    this->_tree.clear();
    this->_copies.clear();
  }

  bool leq(const SeparateDomain& other) const override {
    if (this->is_bottom()) {
      return true;
    } else if (other.is_bottom()) {
      return false;
    } else {
      if (!this->_tree.leq(other._tree, [](const Value& x, const Value& y) {
            return x.leq(y);
          })) {
        return false;
      }
      for (const auto& entry : other._copies) {
        if (entry.first != *entry.second.begin()) {
          continue;
        }
        ClassKey key = this->class_key(entry.first);
        for (VariableRef member : entry.second) {
          if (this->class_key(member) != key) {
            return false;
          }
        }
      }
      return true;
    }
  }

  bool equals(const SeparateDomain& other) const override {
    if (this->is_bottom()) {
      return other.is_bottom();
    } else if (other.is_bottom()) {
      return false;
    } else {
      return this->_tree.equals(other._tree,
                                [](const Value& x, const Value& y) {
                                  return x.equals(y);
                                }) &&
             this->_copies.equals(other._copies,
                                  [](const CopySet& x, const CopySet& y) {
                                    return x.equals(y);
                                  });
    }
  }

  void join_with(const SeparateDomain& other) override {
    if (other.is_bottom()) {
      return;
    } else if (this->is_bottom()) {
      this->operator=(other);
    } else {
      this->intersect_copies(other);
      this->_tree.intersect_with(other._tree,
                                 [](const Value& x, const Value& y) {
                                   Value z = x.join(y);
                                   if (z.is_top()) {
                                     return boost::optional< Value >(
                                         boost::none);
                                   }
                                   return boost::optional< Value >(z);
                                 });
      this->reduce_copies();
    }
  }

  void join_loop_with(const SeparateDomain& other) override {
    if (other.is_bottom()) {
      return;
    } else if (this->is_bottom()) {
      this->operator=(other);
    } else {
      this->intersect_copies(other);
      this->_tree.intersect_with(other._tree,
                                 [](const Value& x, const Value& y) {
                                   Value z = x.join_loop(y);
                                   if (z.is_top()) {
                                     return boost::optional< Value >(
                                         boost::none);
                                   }
                                   return boost::optional< Value >(z);
                                 });
      this->reduce_copies();
    }
  }

  void join_iter_with(const SeparateDomain& other) override {
    if (other.is_bottom()) {
      return;
    } else if (this->is_bottom()) {
      this->operator=(other);
    } else {
      this->intersect_copies(other);
      this->_tree.intersect_with(other._tree,
                                 [](const Value& x, const Value& y) {
                                   Value z = x.join_loop(y);
                                   if (z.is_top()) {
                                     return boost::optional< Value >(
                                         boost::none);
                                   }
                                   return boost::optional< Value >(z);
                                 });
      this->reduce_copies();
    }
  }

  void widen_with(const SeparateDomain& other) override {
    if (other.is_bottom()) {
      return;
    } else if (this->is_bottom()) {
      this->operator=(other);
    } else {
      this->intersect_copies(other);
      this->_tree.intersect_with(other._tree,
                                 [](const Value& x, const Value& y) {
                                   Value z = x.widening(y);
                                   if (z.is_top()) {
                                     return boost::optional< Value >(
                                         boost::none);
                                   }
                                   return boost::optional< Value >(z);
                                 });
      this->reduce_copies();
    }
  }

  void widen_threshold_with(const SeparateDomain& other,
                            const MachineInt& threshold) {
    if (other.is_bottom()) {
      return;
    } else if (this->is_bottom()) {
      this->operator=(other);
    } else {
      this->intersect_copies(other);
      this->_tree.intersect_with(other._tree,
                                 [threshold](const Value& x, const Value& y) {
                                   Value z = x.widening_threshold(y, threshold);
                                   if (z.is_top()) {
                                     return boost::optional< Value >(
                                         boost::none);
                                   }
                                   return boost::optional< Value >(z);
                                 });
      this->reduce_copies();
    }
  }

  void meet_with(const SeparateDomain& other) override {
    if (this->is_bottom()) {
      return;
    } else if (other.is_bottom()) {
      this->set_to_bottom();
    } else {
      try {
        this->_tree.join_with(other._tree, [](const Value& x, const Value& y) {
          Value z = x.meet(y);
          if (z.is_bottom()) {
            throw BottomFound();
          }
          return boost::optional< Value >(z);
        });
      } catch (BottomFound&) {
        this->set_to_bottom();
      }
      if (!this->is_bottom()) {
        this->meet_copies(other);
      }
    }
  }

  void narrow_with(const SeparateDomain& other) override {
    if (this->is_bottom()) {
      return;
    } else if (other.is_bottom()) {
      this->set_to_bottom();
    } else {
      try {
        this->_tree.join_with(other._tree, [](const Value& x, const Value& y) {
          Value z = x.narrowing(y);
          if (z.is_bottom()) {
            throw BottomFound();
          }
          return boost::optional< Value >(z);
        });
      } catch (BottomFound&) {
        this->set_to_bottom();
      }
      if (!this->is_bottom()) {
        this->meet_copies(other);
      }
    }
  }

  void narrow_threshold_with(const SeparateDomain& other,
                             const MachineInt& threshold) {
    if (this->is_bottom()) {
      return;
    } else if (other.is_bottom()) {
      this->set_to_bottom();
    } else {
      try {
        this->_tree.join_with(other._tree,
                              [threshold](const Value& x, const Value& y) {
                                Value z = x.narrowing_threshold(y, threshold);
                                if (z.is_bottom()) {
                                  throw BottomFound();
                                }
                                return boost::optional< Value >(z);
                              });
      } catch (BottomFound&) {
        this->set_to_bottom();
      }
      if (!this->is_bottom()) {
        this->meet_copies(other);
      }
    }
  }

  /// \brief Get the abstract value for the given variable
  Value get(VariableRef x) const {
    if (this->is_bottom()) {
      return Value::bottom(VariableTrait::bit_width(x), VariableTrait::sign(x));
    } else {
      boost::optional< const Value& > v = this->_tree.at(x);
      if (v) {
        return *v;
      } else {
        return Value::top(VariableTrait::bit_width(x), VariableTrait::sign(x));
      }
    }
  }

  /// \brief Set the abstract value of the given variable
  void set(VariableRef x, const Value& value) {
    if (this->is_bottom()) {
      return;
    }
    this->forget_copy(x);
    this->set_value(x, value);
  }

  /// \brief Refine all unchanged copies of the given variable
  void refine(VariableRef x, const Value& value) {
    if (this->is_bottom() || value.is_top()) {
      return;
    }
    Value refined = this->get(x).meet(value);
    CopySet members{x};
    if (auto group = this->_copies.at(x)) {
      members = *group;
    }
    if (refined.is_bottom()) {
      this->set_to_bottom();
      return;
    }
    for (VariableRef member : members) {
      this->set_value(member, refined);
      if (refined.singleton()) {
        this->_copies.erase(member);
      }
    }
  }

  /// \brief Projection
  ///
  /// Return an overapproximation of the linear expression e as an abstract
  /// value
  ///
  /// Note that it wraps on integer overflow.
  /// Note that it will automatically cast variables to the type of
  /// `e.constant()`.
  Value project(const LinearExpressionT& e) const {
    // Result type
    uint64_t bit_width = e.constant().bit_width();
    Signedness sign = e.constant().sign();

    if (this->is_bottom()) {
      return Value::bottom(bit_width, sign);
    }

    Value r(e.constant());
    for (const auto& term : e) {
      r = add(r,
              mul(Value(term.second),
                  this->get(term.first).cast(bit_width, sign)));
    }
    return r;
  }

  /// \brief Forget the abstract value of the given variable
  void forget(VariableRef x) {
    if (this->is_bottom()) {
      return;
    }
    this->forget_copy(x);
    this->_tree.erase(x);
  }

  /// \brief Assign `x = n`
  void assign(VariableRef x, const MachineInt& n) { this->set(x, Value(n)); }

  /// \brief Assign `x = y`
  void assign(VariableRef x, VariableRef y) {
    if (this->is_bottom() || x == y) {
      return;
    }
    this->set(x, this->get(y));
    if (VariableTrait::bit_width(x) == VariableTrait::bit_width(y) &&
        VariableTrait::sign(x) == VariableTrait::sign(y)) {
      this->merge_copies(CopySet{x, y});
    }
  }

  /// \brief Assign `x = e`
  ///
  /// Note that it wraps on integer overflow.
  /// Note that it will automatically cast variables to the type of `x`.
  void assign(VariableRef x, const LinearExpressionT& e) {
    this->set(x, this->project(e));
  }

  /// \brief Apply `x = op y`
  void apply(UnaryOperator op, VariableRef x, VariableRef y) {
    this->set(x,
              apply_unary_operator(op,
                                   this->get(y),
                                   VariableTrait::bit_width(x),
                                   VariableTrait::sign(x)));
  }

  /// \brief Apply `x = y op z`
  void apply(BinaryOperator op, VariableRef x, VariableRef y, VariableRef z) {
    this->set(x, apply_bin_operator(op, this->get(y), this->get(z)));
  }

  /// \brief Apply `x = y op z`
  void apply(BinaryOperator op,
             VariableRef x,
             VariableRef y,
             const MachineInt& z) {
    this->set(x, apply_bin_operator(op, this->get(y), Value(z)));
  }

  /// \brief Apply `x = y op z`
  void apply(BinaryOperator op,
             VariableRef x,
             const MachineInt& y,
             VariableRef z) {
    this->set(x, apply_bin_operator(op, Value(y), this->get(z)));
  }

  void dump(std::ostream& o) const override {
    if (this->is_bottom()) {
      o << "⊥";
    } else {
      this->_tree.dump(o);
      if (!this->_copies.empty()) {
        o << " copies={";
        bool first = true;
        for (const auto& entry : this->_copies) {
          if (entry.first == *entry.second.begin()) {
            if (!first) {
              o << "; ";
            }
            first = false;
            entry.second.dump(o);
          }
        }
        o << "}";
      }
    }
  }

  static std::string name() { return "separate domain of " + Value::name(); }

}; // end class SeparateDomain

} // end namespace machine_int
} // end namespace core
} // end namespace ikos
