use std::fmt;

use datafusion::error::DataFusionError;

pub type Result<T> = std::result::Result<T, PromqlError>;

#[derive(Debug)]
pub enum PromqlError {
    /// Error parsing a PromQL query string.
    Parse(String),
    /// Error during logical plan construction.
    Plan(String),
    /// Error from DataFusion execution.
    Execution(DataFusionError),
    /// Error from the data source backend.
    DataSource(String),
    /// Feature not yet implemented.
    NotImplemented(String),
}

impl fmt::Display for PromqlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(msg) => write!(f, "PromQL parse error: {msg}"),
            Self::Plan(msg) => write!(f, "PromQL plan error: {msg}"),
            Self::Execution(e) => write!(f, "PromQL execution error: {e}"),
            Self::DataSource(msg) => write!(f, "PromQL data source error: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl std::error::Error for PromqlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution(e) => Some(e),
            _ => None,
        }
    }
}

impl From<DataFusionError> for PromqlError {
    fn from(e: DataFusionError) -> Self {
        Self::Execution(e)
    }
}
