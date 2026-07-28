//! Typed, runtime-neutral execution telemetry.
//!
//! Observers receive short-lived borrowed events and decide where, if anywhere,
//! to record them. The crate does not install a logger or tracing subscriber.

use std::collections::BTreeMap;
use std::fmt;
use std::time::{Duration, Instant};

use crate::operation::NodeId;
use crate::prepare::TransformSpec;

/// A measurable phase of model-weight execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ExecutionPhase {
    /// Waiting in a bounded work or resource queue.
    QueueWait,
    /// Hashing source or cache content.
    Hashing,
    /// Establishing or retaining a source mapping.
    Mapping,
    /// Reading source bytes that are not retained in a mapping.
    SourceRead,
    /// Converting, decoding, or repacking host bytes.
    Transform,
    /// The complete caller-supplied preparation callback.
    Preparation,
    /// The consumer-owned delivery callback.
    DeliveryCallback,
    /// Looking up and validating a cache entry.
    CacheLookup,
}

/// Classifies one executed tensor operation for profiling.
///
/// Kinds describe work selected by an already-validated binding plan. A
/// [`Self::Cast`] event therefore reports an executed dtype conversion; it
/// does not imply that every checkpoint requires that conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum OperationKind {
    /// The source representation was retained without materializing bytes.
    Identity,
    /// A provider changed the scalar dtype.
    Cast,
    /// Multiple tensors were concatenated along one axis.
    Concat,
    /// Tensor axes or physical storage were permuted.
    Permute,
    /// A materialized slice was selected.
    Slice,
    /// One tensor was split into ordered materialized outputs.
    Split,
    /// Contiguous bytes were reinterpreted under a new logical shape.
    Reshape,
    /// A provider preparation that is not classified as a dtype cast.
    Prepare,
}

impl OperationKind {
    /// Returns a stable snake-case label for reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Cast => "cast",
            Self::Concat => "concat",
            Self::Permute => "permute",
            Self::Slice => "slice",
            Self::Split => "split",
            Self::Reshape => "reshape",
            Self::Prepare => "prepare",
        }
    }

    pub(crate) fn for_transform(transform: &TransformSpec) -> Self {
        if transform.source().dtype() == transform.target().dtype() {
            Self::Prepare
        } else {
            Self::Cast
        }
    }
}

/// Locates an operation within one materialized work item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum OperationLocation {
    /// A binding whose selected source already satisfies the target.
    Binding,
    /// One transform from a target's ordered preparation chain.
    PlannedTransform {
        /// Zero-based index in the target transform chain.
        index: usize,
    },
    /// One topologically ordered operation-graph node.
    GraphNode(NodeId),
}

/// A bounded host-memory resource class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum MemoryKind {
    /// Bytes read from sources outside retained mappings.
    Source,
    /// Temporary bytes used while preparing a representation.
    Scratch,
    /// Prepared bytes retained until consumer delivery completes.
    Prepared,
}

/// A typed progress event emitted during execution.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionEvent {
    /// A pipeline invocation started.
    Started {
        /// Number of submitted work items.
        work_items: usize,
        /// Number of host workers started.
        workers: usize,
    },
    /// A work item entered the bounded worker queue.
    WorkQueued {
        /// Stable delivery ordinal.
        ordinal: u64,
    },
    /// A measured phase started.
    PhaseStarted {
        /// Phase being measured.
        phase: ExecutionPhase,
        /// Work-item ordinal, when the phase belongs to one item.
        ordinal: Option<u64>,
    },
    /// A measured phase completed.
    PhaseFinished {
        /// Phase that completed.
        phase: ExecutionPhase,
        /// Work-item ordinal, when the phase belongs to one item.
        ordinal: Option<u64>,
        /// Time spent in the phase.
        duration: Duration,
        /// Bytes attributed to the phase.
        bytes: u64,
    },
    /// One host tensor operation completed successfully.
    ///
    /// Input and output byte counts describe tensor spans, not measured memory
    /// traffic. Sums saturate at [`u64::MAX`]. `materialized_output_bytes` is
    /// zero when the operation retained an existing immutable byte view.
    OperationFinished {
        /// Stable pipeline work ordinal, when execution belongs to a pipeline.
        work_ordinal: Option<u64>,
        /// Binding, transform, or graph-node identity within the work item.
        location: OperationLocation,
        /// Executed operation category.
        kind: OperationKind,
        /// Time spent executing this operation.
        duration: Duration,
        /// Sum of ordered input tensor byte lengths.
        input_bytes: u64,
        /// Sum of ordered output tensor byte lengths.
        output_bytes: u64,
        /// Output bytes allocated and populated by this operation.
        materialized_output_bytes: u64,
    },
    /// A weighted memory reservation changed.
    BudgetUsage {
        /// Resource class that changed.
        kind: MemoryKind,
        /// Bytes currently reserved.
        used: u64,
        /// Configured byte limit.
        limit: u64,
    },
    /// The out-of-order delivery queue depth changed.
    DeliveryQueueDepth {
        /// Completed items waiting for an earlier ordinal.
        queued: usize,
        /// Configured completed-delivery queue limit.
        limit: usize,
    },
    /// A work item completed host preparation.
    WorkPrepared {
        /// Stable delivery ordinal.
        ordinal: u64,
        /// Actual prepared bytes returned by the callback.
        bytes: u64,
    },
    /// A prepared item was accepted by the consumer sink.
    WorkDelivered {
        /// Stable delivery ordinal.
        ordinal: u64,
        /// Prepared bytes transferred to the sink.
        bytes: u64,
    },
    /// Cooperative cancellation was observed.
    Cancelled,
    /// A pipeline invocation finished.
    Finished {
        /// Whether every submitted item was delivered.
        success: bool,
        /// End-to-end invocation duration.
        wall_time: Duration,
    },
}

/// Receives typed execution events without imposing a telemetry runtime.
///
/// Implementations should return quickly because worker threads invoke the
/// observer synchronously. Expensive exporters can enqueue copied events into
/// their own bounded channel.
pub trait ExecutionObserver: Send + Sync {
    /// Observes one execution event.
    fn observe(&self, event: &ExecutionEvent);

    /// Returns whether operation-level timing and events are requested.
    ///
    /// Operation events are disabled by default to avoid timers on graph and
    /// transform hot paths. Implementations can override this method, or use
    /// [`with_operation_events`] to opt in.
    fn operation_events_enabled(&self) -> bool {
        false
    }
}

impl<F> ExecutionObserver for F
where
    F: Fn(&ExecutionEvent) + Send + Sync,
{
    fn observe(&self, event: &ExecutionEvent) {
        self(event);
    }
}

/// An observer adapter that enables operation-level timing and events.
///
/// Use [`with_operation_events`] to construct this adapter, including around
/// closure observers covered by the blanket [`ExecutionObserver`]
/// implementation.
pub struct OperationEvents<O> {
    observer: O,
}

impl<O> OperationEvents<O> {
    /// Returns the wrapped observer.
    #[must_use]
    pub fn into_inner(self) -> O {
        self.observer
    }
}

impl<O> fmt::Debug for OperationEvents<O> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OperationEvents")
            .finish_non_exhaustive()
    }
}

impl<O> ExecutionObserver for OperationEvents<O>
where
    O: ExecutionObserver,
{
    fn observe(&self, event: &ExecutionEvent) {
        self.observer.observe(event);
    }

    fn operation_events_enabled(&self) -> bool {
        true
    }
}

/// Enables operation-level timing and events for `observer`.
///
/// Coarse execution events remain enabled for ordinary observers, while
/// operation events require this explicit opt-in.
///
/// # Examples
///
/// ```
/// use model_weights::telemetry::{
///     ExecutionEvent, ExecutionObserver, with_operation_events,
/// };
///
/// let observer = with_operation_events(|_event: &ExecutionEvent| {});
/// assert!(observer.operation_events_enabled());
/// ```
#[must_use]
pub fn with_operation_events<O>(observer: O) -> OperationEvents<O>
where
    O: ExecutionObserver,
{
    OperationEvents { observer }
}

/// An observer that intentionally discards every event.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopObserver;

impl ExecutionObserver for NoopObserver {
    fn observe(&self, _event: &ExecutionEvent) {}
}

/// Counts completed pipeline actions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecutionCounters {
    submitted: u64,
    prepared: u64,
    delivered: u64,
    failed: u64,
    delivered_bytes: u64,
}

impl ExecutionCounters {
    /// Returns the number of submitted work items.
    #[must_use]
    pub const fn submitted(&self) -> u64 {
        self.submitted
    }

    /// Returns the number of successfully prepared work items.
    #[must_use]
    pub const fn prepared(&self) -> u64 {
        self.prepared
    }

    /// Returns the number of sink-accepted work items.
    #[must_use]
    pub const fn delivered(&self) -> u64 {
        self.delivered
    }

    /// Returns the number of preparation or delivery failures observed.
    #[must_use]
    pub const fn failed(&self) -> u64 {
        self.failed
    }

    /// Returns the prepared bytes accepted by the sink.
    #[must_use]
    pub const fn delivered_bytes(&self) -> u64 {
        self.delivered_bytes
    }
}

/// Peak weighted bytes reserved by a pipeline invocation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PeakBytes {
    source: u64,
    scratch: u64,
    prepared: u64,
}

impl PeakBytes {
    /// Returns peak source-byte reservations.
    #[must_use]
    pub const fn source(&self) -> u64 {
        self.source
    }

    /// Returns peak transform-scratch reservations.
    #[must_use]
    pub const fn scratch(&self) -> u64 {
        self.scratch
    }

    /// Returns peak prepared-byte reservations.
    #[must_use]
    pub const fn prepared(&self) -> u64 {
        self.prepared
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{ExecutionEvent, ExecutionObserver, with_operation_events};

    #[test]
    fn closure_observers_skip_operation_events_by_default() {
        let observer = |_event: &ExecutionEvent| {};

        assert!(!observer.operation_events_enabled());
    }

    #[test]
    fn operation_event_adapter_enables_and_forwards_events() {
        let event_count = AtomicU64::new(0);
        let observer = with_operation_events(|_event: &ExecutionEvent| {
            event_count.fetch_add(1, Ordering::Relaxed);
        });

        assert!(observer.operation_events_enabled());
        observer.observe(&ExecutionEvent::Cancelled);
        assert_eq!(event_count.load(Ordering::Relaxed), 1);
    }
}

/// Aggregated timing, count, memory, and throughput measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReport {
    wall_time: Duration,
    phase_durations: BTreeMap<ExecutionPhase, Duration>,
    counters: ExecutionCounters,
    peak_bytes: PeakBytes,
    peak_delivery_queue_depth: usize,
}

impl ExecutionReport {
    /// Returns end-to-end wall time.
    #[must_use]
    pub const fn wall_time(&self) -> Duration {
        self.wall_time
    }

    /// Returns aggregate time attributed to one phase.
    #[must_use]
    pub fn phase_duration(&self, phase: ExecutionPhase) -> Duration {
        self.phase_durations
            .get(&phase)
            .copied()
            .unwrap_or_default()
    }

    /// Iterates over phases that recorded nonzero or explicitly measured time.
    #[must_use]
    pub fn phase_durations(
        &self,
    ) -> impl ExactSizeIterator<Item = (ExecutionPhase, Duration)> + '_ {
        self.phase_durations
            .iter()
            .map(|(phase, duration)| (*phase, *duration))
    }

    /// Returns completed-action counters.
    #[must_use]
    pub const fn counters(&self) -> &ExecutionCounters {
        &self.counters
    }

    /// Returns peak weighted memory reservations.
    #[must_use]
    pub const fn peak_bytes(&self) -> PeakBytes {
        self.peak_bytes
    }

    /// Returns the peak completed items waiting for an earlier ordinal.
    #[must_use]
    pub const fn peak_delivery_queue_depth(&self) -> usize {
        self.peak_delivery_queue_depth
    }

    /// Returns delivered prepared bytes per wall-clock second.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "throughput is an approximate telemetry value"
    )]
    pub fn throughput_bytes_per_second(&self) -> f64 {
        let seconds = self.wall_time.as_secs_f64();
        if seconds > 0.0 {
            self.counters.delivered_bytes as f64 / seconds
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub(crate) struct ExecutionReportBuilder {
    started: Instant,
    phase_durations: BTreeMap<ExecutionPhase, Duration>,
    counters: ExecutionCounters,
    peak_delivery_queue_depth: usize,
}

impl ExecutionReportBuilder {
    pub(crate) fn new(submitted: usize) -> Self {
        Self {
            started: Instant::now(),
            phase_durations: BTreeMap::new(),
            counters: ExecutionCounters {
                submitted: u64::try_from(submitted).unwrap_or(u64::MAX),
                ..ExecutionCounters::default()
            },
            peak_delivery_queue_depth: 0,
        }
    }

    pub(crate) fn add_phase(&mut self, phase: ExecutionPhase, duration: Duration) {
        let total = self.phase_durations.entry(phase).or_default();
        *total = total.saturating_add(duration);
    }

    pub(crate) const fn prepared(&mut self) {
        self.counters.prepared = self.counters.prepared.saturating_add(1);
    }

    pub(crate) const fn delivered(&mut self, bytes: u64) {
        self.counters.delivered = self.counters.delivered.saturating_add(1);
        self.counters.delivered_bytes = self.counters.delivered_bytes.saturating_add(bytes);
    }

    pub(crate) const fn failed(&mut self) {
        self.counters.failed = self.counters.failed.saturating_add(1);
    }

    pub(crate) const fn observe_delivery_queue_depth(&mut self, depth: usize) {
        if depth > self.peak_delivery_queue_depth {
            self.peak_delivery_queue_depth = depth;
        }
    }

    pub(crate) fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub(crate) fn finish(self, peaks: PeakBytes) -> ExecutionReport {
        ExecutionReport {
            wall_time: self.started.elapsed(),
            phase_durations: self.phase_durations,
            counters: self.counters,
            peak_bytes: peaks,
            peak_delivery_queue_depth: self.peak_delivery_queue_depth,
        }
    }
}

impl PeakBytes {
    pub(crate) const fn new(source: u64, scratch: u64, prepared: u64) -> Self {
        Self {
            source,
            scratch,
            prepared,
        }
    }
}
