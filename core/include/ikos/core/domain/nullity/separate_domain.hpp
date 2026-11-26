/*******************************************************************************
 *
 * \file
 * \brief Implementation of an abstract domain keeping track of null variables
 * using a separate domain.
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
#include <utility>

#include <ikos/core/adt/patricia_tree/set.hpp>
#include <ikos/core/domain/nullity/abstract_domain.hpp>
#include <ikos/core/domain/separate_domain.hpp>

namespace ikos {
namespace core {
namespace nullity {

/// \brief Nullity abstract domain
///
/// A separate domain augmented with equal-nullity classes established by
/// copies. Refining a loaded pointer also refines its unchanged memory cell.
template < typename VariableRef >
class SeparateDomain final
    : public nullity::AbstractDomain< VariableRef,
                                      SeparateDomain< VariableRef > > {
private:
  using SeparateDomainT = core::SeparateDomain< VariableRef, Nullity >;
  using CopySet = PatriciaTreeSet< VariableRef >;
  using CopyMap = PatriciaTreeMap< VariableRef, CopySet >;
  using ClassKey = std::pair< unsigned, Index >;

public:
  using Iterator = typename SeparateDomainT::Iterator;

private:
  SeparateDomainT _inv;

  // Only classes of at least two variables with unknown nullity are stored.
  // Every member maps to the complete class, so forgetting one member keeps
  // the relation between the others. The persistent containers keep copies
  // of an abstract state cheap and independent.
  CopyMap _copies;

  ClassKey class_key(VariableRef x) const {
    Nullity value = this->_inv.get(x);
    if (value.is_null()) {
      return {1, 0};
    } else if (value.is_non_null()) {
      return {2, 0};
    }
    auto members = this->_copies.at(x);
    return {0,
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

  void merge_copies(CopySet members) {
    CopySet all = members;
    for (VariableRef member : members) {
      if (auto group = this->_copies.at(member)) {
        all.join_with(*group);
      }
    }
    Nullity value = Nullity::top();
    for (VariableRef member : all) {
      value.meet_with(this->_inv.get(member));
    }
    if (value.is_bottom()) {
      this->set_to_bottom();
      return;
    }
    for (VariableRef member : all) {
      this->_inv.set(member, value);
      this->_copies.erase(member);
    }
    if (value.is_top() && all.size() > 1) {
      for (VariableRef member : all) {
        this->_copies.insert_or_assign(member, all);
      }
    }
  }

private:
  /// \brief Private constructor
  explicit SeparateDomain(SeparateDomainT inv) : _inv(std::move(inv)) {}

public:
  /// \brief Create the top abstract value
  static SeparateDomain top() { return SeparateDomain(SeparateDomainT::top()); }

  /// \brief Create the bottom abstract value
  static SeparateDomain bottom() {
    return SeparateDomain(SeparateDomainT::bottom());
  }

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

  /// \brief Begin iterator over the pairs (variable, nullity)
  Iterator begin() const { return this->_inv.begin(); }

  /// \brief End iterator over the pairs (variable, nullity)
  Iterator end() const { return this->_inv.end(); }

  void normalize() override {}

  bool is_bottom() const override { return this->_inv.is_bottom(); }

  bool is_top() const override {
    return this->_inv.is_top() && this->_copies.empty();
  }

  void set_to_bottom() override {
    this->_inv.set_to_bottom();
    this->_copies.clear();
  }

  void set_to_top() override {
    this->_inv.set_to_top();
    this->_copies.clear();
  }

  bool leq(const SeparateDomain& other) const override {
    if (this->is_bottom()) {
      return true;
    } else if (!this->_inv.leq(other._inv)) {
      return false;
    }
    for (const auto& entry : other._copies) {
      VariableRef representative = *entry.second.begin();
      if (entry.first != representative) {
        continue;
      }
      ClassKey key = this->class_key(representative);
      for (VariableRef member : entry.second) {
        if (this->class_key(member) != key) {
          return false;
        }
      }
    }
    return true;
  }

  bool equals(const SeparateDomain& other) const override {
    return this->_inv.equals(other._inv) &&
           this->_copies.equals(other._copies,
                                [](const CopySet& x, const CopySet& y) {
                                  return x.equals(y);
                                });
  }

  void join_with(const SeparateDomain& other) override {
    if (this == &other || other.is_bottom()) {
      return;
    } else if (this->is_bottom()) {
      *this = other;
      return;
    }

    // Intersect the two equivalence relations. Definite nullities also imply
    // equal-nullity constraints, even if no explicit copy class is stored.
    CopySet variables;
    for (const auto& entry : this->_inv) {
      variables.insert(entry.first);
    }
    for (const auto& entry : other._inv) {
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
      if (this->_inv.get(variable).join(other._inv.get(variable)).is_top()) {
        groups[{this->class_key(variable), other.class_key(variable)}].insert(
            variable);
      }
    }
    this->_inv.join_with(other._inv);
    this->_copies.clear();
    for (const auto& entry : groups) {
      if (entry.second.size() > 1) {
        for (VariableRef member : entry.second) {
          this->_copies.insert_or_assign(member, entry.second);
        }
      }
    }
  }

  void widen_with(const SeparateDomain& other) override {
    this->join_with(other);
  }

  void meet_with(const SeparateDomain& other) override {
    if (this == &other || this->is_bottom()) {
      return;
    } else if (other.is_bottom()) {
      this->set_to_bottom();
      return;
    }
    this->_inv.meet_with(other._inv);
    if (this->is_bottom()) {
      this->_copies.clear();
      return;
    }
    CopyMap copies = this->_copies;
    this->_copies.clear();
    for (const auto& entry : copies) {
      if (entry.first == *entry.second.begin()) {
        this->merge_copies(entry.second);
        if (this->is_bottom()) {
          return;
        }
      }
    }
    for (const auto& entry : other._copies) {
      if (entry.first == *entry.second.begin()) {
        this->merge_copies(entry.second);
        if (this->is_bottom()) {
          return;
        }
      }
    }
  }

  void narrow_with(const SeparateDomain& other) override {
    this->meet_with(other);
  }

  void assign_null(VariableRef x) override { this->set(x, Nullity::null()); }

  void assign_non_null(VariableRef x) override {
    this->set(x, Nullity::non_null());
  }

  void assign(VariableRef x, VariableRef y) override {
    if (this->is_bottom() || x == y) {
      return;
    }
    Nullity value = this->_inv.get(y);
    this->set(x, value);
    if (value.is_top()) {
      this->merge_copies(CopySet{x, y});
    }
  }

  void assert_null(VariableRef x) override { this->refine(x, Nullity::null()); }

  void assert_non_null(VariableRef x) override {
    this->refine(x, Nullity::non_null());
  }

  void add(Predicate pred, VariableRef x, VariableRef y) override {
    if (this->is_bottom()) {
      return;
    }

    Nullity xn = this->_inv.get(x);
    Nullity yn = this->_inv.get(y);

    switch (pred) {
      case Predicate::EQ: {
        // x == y
        this->merge_copies(CopySet{x, y});
      } break;
      case Predicate::NE: {
        // x != y
        if (xn.is_null() && yn.is_null()) {
          this->set_to_bottom();
        } else if (xn.is_top() && yn.is_null()) {
          this->refine(x, Nullity::non_null());
        } else if (xn.is_null() && yn.is_top()) {
          this->refine(y, Nullity::non_null());
        }
      } break;
      case Predicate::GT: {
        this->add(Predicate::NE, x, y);
      } break;
      case Predicate::GE: {
        // nothing we can do.
      } break;
      case Predicate::LT: {
        this->add(Predicate::NE, x, y);
      } break;
      case Predicate::LE: {
        // nothing we can do.
      } break;
    }
  }

  bool is_null(VariableRef x) const override {
    Nullity value = this->_inv.get(x);
    return value.is_bottom() || value.is_null();
  }

  bool is_non_null(VariableRef x) const override {
    Nullity value = this->_inv.get(x);
    return value.is_bottom() || value.is_non_null();
  }

  void set(VariableRef x, const Nullity& value) override {
    if (this->is_bottom()) {
      return;
    }
    this->forget_copy(x);
    this->_inv.set(x, value);
    if (this->is_bottom()) {
      this->_copies.clear();
    }
  }

  void refine(VariableRef x, const Nullity& value) override {
    if (this->is_bottom()) {
      return;
    }
    Nullity refined = this->_inv.get(x).meet(value);
    if (refined.is_bottom()) {
      this->set_to_bottom();
    } else if (!refined.is_top()) {
      CopySet members{x};
      if (auto group = this->_copies.at(x)) {
        members = *group;
      }
      for (VariableRef member : members) {
        this->_inv.set(member, refined);
        this->_copies.erase(member);
      }
    }
  }

  void forget(VariableRef x) override {
    this->forget_copy(x);
    this->_inv.forget(x);
  }

  Nullity get(VariableRef x) const override { return this->_inv.get(x); }

  void dump(std::ostream& o) const override {
    this->_inv.dump(o);
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

  static std::string name() { return "nullity domain"; }

}; // end class SeparateDomain

} // end namespace nullity
} // end namespace core
} // end namespace ikos
