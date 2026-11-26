#pragma once

namespace ikos {
namespace core {
namespace machine_int {
class IntervalCongruence;
}
} // namespace core
namespace ar {
class Load;
}
namespace analyzer {
class CallContext;

/// Prove initialization on every defined path to a scalar load, through
/// complete stores or zeroed allocations in the actual calling context.
bool prove_initialized_load(ar::Load* load, CallContext* context);

/// Prove that every defined path reads an untouched malloc allocation.
/// Failure means unknown, never initialized.
bool prove_uninitialized_load(ar::Load* load, CallContext* context);

/// Prove bounded, completed loop stores cover every possible load offset.
/// Only nonescaping stack objects and verified counted loops are supported.
bool prove_loop_initialized_load(
    ar::Load* load, const core::machine_int::IntervalCongruence& offset);

} // namespace analyzer
} // namespace ikos
