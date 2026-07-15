//! Backpressure contracts and bounded transfer queues.

use crate::{DistributeError, DistributeResult, NamedTransfer};

/// Signal indicating consumer pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackpressureSignal {
    /// Accept more work.
    Ok,
    /// Soft warn at high watermark.
    SoftLimit,
    /// Hard reject.
    HardLimit,
}

/// Watermark policy for explicit backpressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BackpressurePolicy {
    /// Soft watermark.
    pub soft_limit: usize,
    /// Hard watermark.
    pub hard_limit: usize,
}

impl BackpressurePolicy {
    /// Creates a validated policy.
    pub fn try_new(soft_limit: usize, hard_limit: usize) -> DistributeResult<Self> {
        if soft_limit == 0 || hard_limit < soft_limit {
            return Err(DistributeError::InvalidConfiguration(
                "require 0 < soft_limit <= hard_limit".into(),
            ));
        }
        Ok(Self {
            soft_limit,
            hard_limit,
        })
    }

    /// Evaluates queue depth against watermarks.
    #[must_use]
    pub fn evaluate(self, depth: usize) -> BackpressureSignal {
        if depth >= self.hard_limit {
            BackpressureSignal::HardLimit
        } else if depth >= self.soft_limit {
            BackpressureSignal::SoftLimit
        } else {
            BackpressureSignal::Ok
        }
    }
}

/// FIFO queue of named transfers with watermark-driven admissions.
#[derive(Clone, Debug)]
pub struct BoundedTransferQueue {
    policy: BackpressurePolicy,
    items: Vec<NamedTransfer>,
    soft_trips: u64,
    hard_rejects: u64,
}

impl BoundedTransferQueue {
    /// Creates an empty queue.
    #[must_use]
    pub fn new(policy: BackpressurePolicy) -> Self {
        Self {
            policy,
            items: Vec::new(),
            soft_trips: 0,
            hard_rejects: 0,
        }
    }

    /// Current depth.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.items.len()
    }

    /// Soft-limit trip count.
    #[must_use]
    pub fn soft_trips(&self) -> u64 {
        self.soft_trips
    }

    /// Hard-reject count.
    #[must_use]
    pub fn hard_rejects(&self) -> u64 {
        self.hard_rejects
    }

    /// Current pressure signal for `depth()`.
    #[must_use]
    pub fn signal(&self) -> BackpressureSignal {
        self.policy.evaluate(self.depth())
    }

    /// Attempts to enqueue a transfer; hard-limit depths are rejected.
    pub fn try_push(&mut self, transfer: NamedTransfer) -> DistributeResult<BackpressureSignal> {
        let signal = self.policy.evaluate(self.depth());
        match signal {
            BackpressureSignal::HardLimit => {
                self.hard_rejects += 1;
                Err(DistributeError::CapacityExceeded {
                    queue: transfer.name,
                    depth: self.depth(),
                    hard_limit: self.policy.hard_limit,
                })
            }
            BackpressureSignal::SoftLimit => {
                self.soft_trips += 1;
                self.items.push(transfer);
                Ok(BackpressureSignal::SoftLimit)
            }
            BackpressureSignal::Ok => {
                self.items.push(transfer);
                Ok(self.policy.evaluate(self.depth()))
            }
        }
    }

    /// Pops the oldest transfer.
    pub fn pop(&mut self) -> Option<NamedTransfer> {
        if self.items.is_empty() {
            None
        } else {
            Some(self.items.remove(0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BackpressurePolicy, BackpressureSignal, BoundedTransferQueue};
    use crate::{NamedTransfer, TransferDirection, TransferKind};

    fn sample(name: &str) -> NamedTransfer {
        NamedTransfer::try_new(
            name,
            TransferDirection::HostToNetwork,
            TransferKind::ExplicitCopy,
            "a",
            "b",
            8,
        )
        .unwrap()
    }

    #[test]
    fn soft_and_hard_limits() {
        let policy = BackpressurePolicy::try_new(1, 2).unwrap();
        let mut queue = BoundedTransferQueue::new(policy);
        assert_eq!(
            queue.try_push(sample("t0")).unwrap(),
            BackpressureSignal::SoftLimit
        );
        assert_eq!(
            queue.try_push(sample("t1")).unwrap(),
            BackpressureSignal::SoftLimit
        );
        assert!(queue.try_push(sample("t2")).is_err());
        assert_eq!(queue.hard_rejects(), 1);
        assert_eq!(queue.pop().unwrap().name, "t0");
    }
}
