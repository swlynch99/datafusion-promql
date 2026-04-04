use std::fmt;

/// Sort functions that reorder vector elements without transforming values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum SortFunction {
    /// Sort by sample value ascending.
    Sort,
    /// Sort by sample value descending.
    SortDesc,
    /// Sort by label values ascending (lexicographic).
    SortByLabel { labels: Vec<String> },
    /// Sort by label values descending (lexicographic).
    SortByLabelDesc { labels: Vec<String> },
}

impl fmt::Display for SortFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sort => write!(f, "sort"),
            Self::SortDesc => write!(f, "sort_desc"),
            Self::SortByLabel { labels } => write!(f, "sort_by_label({})", labels.join(", ")),
            Self::SortByLabelDesc { labels } => {
                write!(f, "sort_by_label_desc({})", labels.join(", "))
            }
        }
    }
}

/// Look up a sort function by name.
///
/// Returns `None` if the name is not a known sort function.
pub(crate) fn lookup_sort_function(name: &str) -> Option<SortFunctionKind> {
    match name {
        "sort" => Some(SortFunctionKind::Sort),
        "sort_desc" => Some(SortFunctionKind::SortDesc),
        "sort_by_label" => Some(SortFunctionKind::SortByLabel),
        "sort_by_label_desc" => Some(SortFunctionKind::SortByLabelDesc),
        _ => None,
    }
}

/// Intermediate kind used during lookup before label args are resolved.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SortFunctionKind {
    Sort,
    SortDesc,
    SortByLabel,
    SortByLabelDesc,
}

impl SortFunctionKind {
    /// Resolve into a full `SortFunction` given the string arguments from the call.
    pub fn resolve(self, label_args: Vec<String>) -> SortFunction {
        match self {
            Self::Sort => SortFunction::Sort,
            Self::SortDesc => SortFunction::SortDesc,
            Self::SortByLabel => SortFunction::SortByLabel { labels: label_args },
            Self::SortByLabelDesc => SortFunction::SortByLabelDesc { labels: label_args },
        }
    }

    /// Whether this kind requires label arguments.
    pub fn requires_labels(self) -> bool {
        matches!(self, Self::SortByLabel | Self::SortByLabelDesc)
    }
}
