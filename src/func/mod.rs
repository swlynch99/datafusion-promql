pub(crate) mod aggregate;
pub(crate) mod instant;
pub(crate) mod range;

pub(crate) use aggregate::{AggregateFunction, lookup_aggregate_function};
pub(crate) use instant::{InstantFunction, lookup_instant_function};
pub(crate) use range::{RangeFunction, lookup_range_function};
