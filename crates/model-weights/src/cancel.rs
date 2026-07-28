use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{Error, Result};

/// A cloneable cooperative-cancellation signal.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates an uncancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation for every clone of this token.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Returns a cancellation error when cancellation was requested.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCategory::Cancelled`](crate::ErrorCategory::Cancelled)
    /// after [`cancel`](Self::cancel) is called on any clone.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::cancelled())
        } else {
            Ok(())
        }
    }
}
