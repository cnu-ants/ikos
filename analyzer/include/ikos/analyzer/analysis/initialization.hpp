#pragma once

namespace ikos {
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

} // namespace analyzer
} // namespace ikos
