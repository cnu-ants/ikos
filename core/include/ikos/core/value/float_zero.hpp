/*******************************************************************************
 *
 * \file
 * \brief Floating point zero classification
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

#include <cstdint>
#include <limits>
#include <ostream>
#include <string>

#include <ikos/core/domain/abstract_domain.hpp>

namespace ikos {
namespace core {

/// Zero/subnormal/other classification, tagged with the stored FP format.
/// "Other" includes normal values, infinities and NaNs. This is not a numeric
/// interval and does not justify any floating arithmetic simplification.
/// A distinct format tag records bytes known to be entirely zero.
class FloatZero final : public AbstractDomain< FloatZero > {
private:
  enum : unsigned { Zero = 1, Subnormal = 2, Other = 4, All = 7 };
  uint64_t _format;
  unsigned _classes;

  FloatZero(uint64_t format, unsigned classes)
      : _format(classes == 0 ? 0 : format), _classes(classes) {}

public:
  static FloatZero top() { return FloatZero(0, All); }
  static FloatZero bottom() { return FloatZero(0, 0); }
  static FloatZero zero(uint64_t format) { return FloatZero(format, Zero); }
  static FloatZero subnormal(uint64_t format) {
    return FloatZero(format, Subnormal);
  }
  static FloatZero other(uint64_t format) { return FloatZero(format, Other); }
  static FloatZero raw_zero() {
    return zero(std::numeric_limits< uint64_t >::max());
  }

  uint64_t format() const { return this->_format; }
  bool is_raw_zero() const {
    return this->_format == std::numeric_limits< uint64_t >::max() &&
           this->is_zero();
  }
  bool is_zero() const { return this->_classes == Zero; }
  bool is_non_zero() const {
    return this->_classes != 0 && !(this->_classes & Zero);
  }
  bool may_be_subnormal() const { return this->_classes & Subnormal; }

  void normalize() override {}
  bool is_bottom() const override { return this->_classes == 0; }
  bool is_top() const override {
    return this->_format == 0 && this->_classes == All;
  }
  void set_to_bottom() override { *this = bottom(); }
  void set_to_top() override { *this = top(); }

  bool leq(const FloatZero& other) const override {
    return this->is_bottom() || other.is_top() ||
           (this->_format == other._format &&
            (this->_classes & other._classes) == this->_classes);
  }
  bool equals(const FloatZero& other) const override {
    return this->_format == other._format && this->_classes == other._classes;
  }
  void join_with(const FloatZero& other) override {
    if (this->is_bottom() || other.is_top()) {
      *this = other;
    } else if (other.is_bottom() || this->is_top()) {
      return;
    } else if (this->_format != other._format) {
      this->set_to_top();
    } else {
      this->_classes |= other._classes;
    }
  }
  void meet_with(const FloatZero& other) override {
    if (this->is_top() || other.is_bottom()) {
      *this = other;
    } else if (other.is_top() || this->is_bottom()) {
      return;
    } else if (this->_format != other._format) {
      this->set_to_bottom();
    } else {
      this->_classes &= other._classes;
      if (this->_classes == 0)
        this->_format = 0;
    }
  }
  void widen_with(const FloatZero& other) override { this->join_with(other); }
  void narrow_with(const FloatZero& other) override { this->meet_with(other); }
  void dump(std::ostream& out) const override {
    if (this->is_top())
      out << "T";
    else if (this->is_bottom())
      out << "bottom";
    else
      out << "format=" << this->_format << ",classes=" << this->_classes;
  }
  static std::string name() { return "floating point zero classification"; }
};

} // namespace core
} // namespace ikos
