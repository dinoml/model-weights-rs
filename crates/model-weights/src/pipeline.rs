//! Bounded, deterministic host preparation and consumer delivery.
//!
//! The pipeline owns worker threads and host-side reservations only. A sink
//! receives each prepared item by value and may transfer it into a runtime-owned
//! asynchronous upload queue.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Formatter};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::limits::ExecutionLimits;
use crate::telemetry::{
    ExecutionEvent, ExecutionObserver, ExecutionPhase, ExecutionReport, ExecutionReportBuilder,
    MemoryKind, PeakBytes,
};
use crate::{CancellationToken, Error, Result};

const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Weighted host resources reserved for one work item.
#[allow(
    clippy::struct_field_names,
    reason = "the byte suffix aligns each resource dimension with ExecutionLimits and telemetry"
)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceWeights {
    source_bytes: u64,
    scratch_bytes: u64,
    prepared_bytes: u64,
}

impl ResourceWeights {
    /// Creates resource weights for source, scratch, and prepared bytes.
    #[must_use]
    pub const fn new(source_bytes: u64, scratch_bytes: u64, prepared_bytes: u64) -> Self {
        Self {
            source_bytes,
            scratch_bytes,
            prepared_bytes,
        }
    }

    /// Returns source bytes reserved while the item is prepared.
    #[must_use]
    pub const fn source_bytes(self) -> u64 {
        self.source_bytes
    }

    /// Returns temporary transform bytes reserved while the item is prepared.
    #[must_use]
    pub const fn scratch_bytes(self) -> u64 {
        self.scratch_bytes
    }

    /// Returns prepared bytes reserved through consumer delivery.
    #[must_use]
    pub const fn prepared_bytes(self) -> u64 {
        self.prepared_bytes
    }
}

/// One independently preparable item with a stable delivery ordinal.
#[derive(Debug)]
pub struct WorkItem<T> {
    ordinal: u64,
    value: T,
    resources: ResourceWeights,
}

impl<T> WorkItem<T> {
    /// Creates a work item.
    #[must_use]
    pub const fn new(ordinal: u64, value: T, resources: ResourceWeights) -> Self {
        Self {
            ordinal,
            value,
            resources,
        }
    }

    /// Returns the stable delivery ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// Returns weighted resource requirements.
    #[must_use]
    pub const fn resources(&self) -> ResourceWeights {
        self.resources
    }

    /// Returns the caller-owned work value.
    #[must_use]
    pub const fn value(&self) -> &T {
        &self.value
    }

    /// Consumes the wrapper and returns the caller-owned work value.
    #[must_use]
    pub fn into_value(self) -> T {
        self.value
    }
}

/// An owned prepared value and its actual host byte size.
#[derive(Debug)]
pub struct PreparedItem<T> {
    value: T,
    bytes: u64,
}

impl<T> PreparedItem<T> {
    /// Creates a prepared item.
    #[must_use]
    pub const fn new(value: T, bytes: u64) -> Self {
        Self { value, bytes }
    }

    /// Returns the actual number of prepared host bytes.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the prepared value.
    #[must_use]
    pub const fn value(&self) -> &T {
        &self.value
    }

    /// Consumes the wrapper and returns the prepared value.
    #[must_use]
    pub fn into_value(self) -> T {
        self.value
    }
}

/// Receives ordinal-ordered prepared values by ownership transfer.
pub trait PreparedSink<T> {
    /// Accepts one prepared item.
    ///
    /// The callback runs on the pipeline coordinator thread. Returning only
    /// after capacity is available supplies backpressure; moving `item` into an
    /// asynchronous runtime queue transfers subsequent memory accounting to the
    /// consumer.
    ///
    /// # Errors
    ///
    /// Returns an error when the consumer rejects the item.
    fn deliver(
        &mut self,
        ordinal: u64,
        item: PreparedItem<T>,
        cancellation: &CancellationToken,
    ) -> Result<()>;
}

impl<T, F> PreparedSink<T> for F
where
    F: FnMut(u64, PreparedItem<T>, &CancellationToken) -> Result<()>,
{
    fn deliver(
        &mut self,
        ordinal: u64,
        item: PreparedItem<T>,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        self(ordinal, item, cancellation)
    }
}

/// Context supplied to a host preparation callback.
pub struct PrepareContext<'a> {
    ordinal: u64,
    cancellation: &'a CancellationToken,
    observer: &'a dyn ExecutionObserver,
    resources: Arc<ResourceBudget>,
    initial_reservation: ResourcePermit,
    reserved: ResourceWeights,
    promoted_prepared: Vec<ResourcePermit>,
    metrics: Vec<PhaseMetric>,
}

impl fmt::Debug for PrepareContext<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrepareContext")
            .field("ordinal", &self.ordinal)
            .field("cancellation", &self.cancellation)
            .field("initial_reservation", &self.initial_reservation)
            .field("reserved", &self.reserved)
            .field("promoted_prepared", &self.promoted_prepared)
            .field("metrics", &self.metrics)
            .finish_non_exhaustive()
    }
}

impl<'a> PrepareContext<'a> {
    fn new(
        ordinal: u64,
        cancellation: &'a CancellationToken,
        observer: &'a dyn ExecutionObserver,
        resources: &Arc<ResourceBudget>,
        initial_reservation: ResourcePermit,
    ) -> Self {
        let reserved = initial_reservation.reserved;
        Self {
            ordinal,
            cancellation,
            observer,
            resources: Arc::clone(resources),
            initial_reservation,
            reserved,
            promoted_prepared: Vec::new(),
            metrics: Vec::new(),
        }
    }

    /// Returns the current work-item ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// Returns the cooperative-cancellation token.
    #[must_use]
    pub const fn cancellation(&self) -> &CancellationToken {
        self.cancellation
    }

    /// Measures a hashing, mapping, read, or transform subphase.
    ///
    /// The operation receives the cancellation token so long-running work can
    /// establish its own prompt yield points.
    pub fn measure<R>(
        &mut self,
        phase: ExecutionPhase,
        bytes: u64,
        operation: impl FnOnce(&CancellationToken) -> R,
    ) -> R {
        self.observer.observe(&ExecutionEvent::PhaseStarted {
            phase,
            ordinal: Some(self.ordinal),
        });
        let started = Instant::now();
        let value = operation(self.cancellation);
        let duration = started.elapsed();
        self.metrics.push(PhaseMetric { phase, duration });
        self.observer.observe(&ExecutionEvent::PhaseFinished {
            phase,
            ordinal: Some(self.ordinal),
            duration,
            bytes,
        });
        value
    }

    /// Records a phase measured by an external implementation.
    pub fn record_phase(&mut self, phase: ExecutionPhase, duration: Duration, bytes: u64) {
        self.metrics.push(PhaseMetric { phase, duration });
        self.observer.observe(&ExecutionEvent::PhaseFinished {
            phase,
            ordinal: Some(self.ordinal),
            duration,
            bytes,
        });
    }

    /// Acquires any additional weighted resources required by a deferred route.
    ///
    /// `required` is the total requirement, including resources already held
    /// for this work item. The callback runs only after the missing dimensions
    /// are acquired atomically. Source and scratch additions are released when
    /// it returns. On success, additional prepared bytes remain reserved until
    /// the prepared item has been delivered.
    pub(crate) fn with_resources<R>(
        &mut self,
        required: ResourceWeights,
        operation: impl FnOnce(&mut Self) -> Result<R>,
    ) -> Result<R> {
        let mut missing = subtract_weights(required, self.reserved);
        if missing == ResourceWeights::default() {
            return operation(self);
        }
        if missing.prepared_bytes > 0 {
            self.initial_reservation.release_prepared();
            self.promoted_prepared.clear();
            self.reserved.prepared_bytes = 0;
            missing = subtract_weights(required, self.reserved);
            emit_budget_usage(&self.resources, self.observer);
        }
        if missing.prepared_bytes > 0 {
            self.promoted_prepared.try_reserve(1).map_err(|_error| {
                Error::limit("could not reserve deferred prepared-permit storage")
            })?;
        }

        self.observer.observe(&ExecutionEvent::PhaseStarted {
            phase: ExecutionPhase::QueueWait,
            ordinal: Some(self.ordinal),
        });
        let started = Instant::now();
        let budget = Arc::clone(&self.resources);
        let reservation = budget.acquire(missing, self.cancellation);
        let duration = started.elapsed();
        self.metrics.push(PhaseMetric {
            phase: ExecutionPhase::QueueWait,
            duration,
        });
        self.observer.observe(&ExecutionEvent::PhaseFinished {
            phase: ExecutionPhase::QueueWait,
            ordinal: Some(self.ordinal),
            duration,
            bytes: 0,
        });
        let reservation = reservation?;
        emit_budget_usage(&budget, self.observer);

        let previous = self.reserved;
        let previous_promotions = self.promoted_prepared.len();
        let (reservation, promoted_prepared) = reservation.split_prepared();
        if let Some(promoted_prepared) = promoted_prepared {
            self.promoted_prepared.push(promoted_prepared);
        }
        self.reserved = max_weights(self.reserved, required);
        let result = operation(self);
        if result.is_ok() {
            let prepared_bytes = self.reserved.prepared_bytes;
            self.reserved = ResourceWeights::new(
                previous.source_bytes,
                previous.scratch_bytes,
                prepared_bytes,
            );
        } else {
            self.promoted_prepared.truncate(previous_promotions);
            self.reserved = previous;
        }
        drop(reservation);
        emit_budget_usage(&budget, self.observer);
        result
    }
}

/// A fixed-width host preparation pipeline.
#[derive(Debug, Clone)]
pub struct Pipeline {
    limits: ExecutionLimits,
    cancellation: CancellationToken,
}

impl Pipeline {
    /// Creates a pipeline with a fresh cancellation token.
    ///
    /// # Errors
    ///
    /// Returns a resource-limit error when any execution limit is zero.
    pub fn new(limits: ExecutionLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            limits,
            cancellation: CancellationToken::new(),
        })
    }

    /// Creates a pipeline using a caller-owned cooperative-cancellation token.
    ///
    /// The token is cancelled when preparation or delivery terminates early so
    /// every in-flight callback observes the same terminal state.
    ///
    /// # Errors
    ///
    /// Returns a resource-limit error when any execution limit is zero.
    pub fn with_cancellation(
        limits: ExecutionLimits,
        cancellation: CancellationToken,
    ) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            limits,
            cancellation,
        })
    }

    /// Returns configured execution limits.
    #[must_use]
    pub const fn limits(&self) -> &ExecutionLimits {
        &self.limits
    }

    /// Returns the shared cancellation token.
    #[must_use]
    pub const fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Prepares work in parallel and delivers it in ascending ordinal order.
    ///
    /// Host worker count, result-channel depth, dispatch lookahead, and weighted
    /// source, scratch, and prepared bytes are bounded independently by
    /// [`ExecutionLimits`].
    ///
    /// # Errors
    ///
    /// Returns an error when ordinals are duplicated, a resource estimate
    /// exceeds its configured limit, preparation or delivery fails, actual
    /// prepared bytes exceed their reservation, or cancellation is observed.
    #[expect(
        clippy::too_many_lines,
        reason = "keeping pipeline setup, scoped workers, and terminal reporting together makes cancellation and teardown auditable"
    )]
    pub fn execute<T, U, P, S, O>(
        &self,
        work: impl IntoIterator<Item = WorkItem<T>>,
        prepare: P,
        sink: &mut S,
        observer: &O,
    ) -> Result<ExecutionReport>
    where
        T: Send,
        U: Send,
        P: Fn(T, &mut PrepareContext<'_>) -> Result<PreparedItem<U>> + Sync,
        S: PreparedSink<U>,
        O: ExecutionObserver,
    {
        self.limits.validate()?;
        let mut work = collect_bounded_work(
            work,
            self.limits.max_work_items,
            &self.cancellation,
            observer,
        )?;
        work.sort_unstable_by_key(WorkItem::ordinal);
        validate_work(&work, &self.limits)?;

        let ordinals = work.iter().map(WorkItem::ordinal).collect::<Vec<_>>();
        let worker_count = self
            .limits
            .workers
            .min(self.limits.dispatch_lookahead)
            .min(work.len());
        let mut report = ExecutionReportBuilder::new(work.len());
        observer.observe(&ExecutionEvent::Started {
            work_items: work.len(),
            workers: worker_count,
        });

        if work.is_empty() {
            return Ok(finish_empty_execution(report, observer));
        }

        if self.cancellation.is_cancelled() {
            observer.observe(&ExecutionEvent::Cancelled);
            observer.observe(&ExecutionEvent::Finished {
                success: false,
                wall_time: report.elapsed(),
            });
            return Err(Error::cancelled());
        }

        let resources = Arc::new(ResourceBudget::new(&self.limits));
        let dispatch_window = Arc::new(DispatchWindow::new(self.limits.dispatch_lookahead));
        let pending_delivery_limit = self.limits.dispatch_lookahead.saturating_sub(1);
        let result = thread::scope(|scope| {
            let (work_sender, work_receiver) = bounded(worker_count);
            let (result_sender, result_receiver) = bounded(self.limits.delivery_queue_depth);

            let dispatcher_cancellation = self.cancellation.clone();
            let dispatcher_observer = observer as &dyn ExecutionObserver;
            let dispatcher_resources = Arc::clone(&resources);
            let dispatcher_window = Arc::clone(&dispatch_window);
            let _dispatcher = scope.spawn(move || {
                dispatch_work(
                    work,
                    &work_sender,
                    &dispatcher_resources,
                    &dispatcher_window,
                    &dispatcher_cancellation,
                    dispatcher_observer,
                );
            });

            for _ in 0..worker_count {
                let worker_receiver = work_receiver.clone();
                let worker_sender = result_sender.clone();
                let worker_resources = Arc::clone(&resources);
                let worker_cancellation = self.cancellation.clone();
                let worker_observer = observer as &dyn ExecutionObserver;
                let prepare = &prepare;
                let _worker = scope.spawn(move || {
                    worker_loop(
                        &worker_receiver,
                        &worker_sender,
                        &worker_resources,
                        &worker_cancellation,
                        worker_observer,
                        prepare,
                    );
                });
            }
            drop(work_receiver);
            drop(result_sender);

            coordinate_results(
                &ordinals,
                &result_receiver,
                sink,
                observer,
                &self.cancellation,
                CoordinatorBounds {
                    resources: &resources,
                    pending_delivery_limit,
                },
                &mut report,
            )
        });

        let peaks = resources.peaks()?;
        match result {
            Ok(()) => {
                let finished = report.finish(peaks);
                observer.observe(&ExecutionEvent::Finished {
                    success: true,
                    wall_time: finished.wall_time(),
                });
                Ok(finished)
            }
            Err(error) => {
                observer.observe(&ExecutionEvent::Finished {
                    success: false,
                    wall_time: report.elapsed(),
                });
                Err(error)
            }
        }
    }
}

fn collect_bounded_work<T>(
    work: impl IntoIterator<Item = WorkItem<T>>,
    maximum_work_items: usize,
    cancellation: &CancellationToken,
    observer: &impl ExecutionObserver,
) -> Result<Vec<WorkItem<T>>> {
    let started = Instant::now();
    let result = (|| {
        cancellation.check()?;
        let mut collected = Vec::new();
        for item in work {
            cancellation.check()?;
            if collected.len() >= maximum_work_items {
                return Err(Error::limit(
                    "pipeline work-item count exceeds the configured limit",
                ));
            }
            collected
                .try_reserve(1)
                .map_err(|_error| Error::limit("could not allocate pipeline work metadata"))?;
            collected.push(item);
        }
        cancellation.check()?;
        Ok(collected)
    })();
    if result.as_ref().is_err_and(Error::is_cancelled) {
        observer.observe(&ExecutionEvent::Cancelled);
        observer.observe(&ExecutionEvent::Finished {
            success: false,
            wall_time: started.elapsed(),
        });
    }
    result
}

fn finish_empty_execution(
    report: ExecutionReportBuilder,
    observer: &impl ExecutionObserver,
) -> ExecutionReport {
    let finished = report.finish(PeakBytes::default());
    observer.observe(&ExecutionEvent::Finished {
        success: true,
        wall_time: finished.wall_time(),
    });
    finished
}

fn validate_work<T>(work: &[WorkItem<T>], limits: &ExecutionLimits) -> Result<()> {
    let mut ordinals = BTreeSet::new();
    for item in work {
        if !ordinals.insert(item.ordinal) {
            return Err(Error::delivery(
                "pipeline work-item ordinals must be unique",
            ));
        }
        let resources = item.resources;
        if resources.source_bytes > limits.source_bytes
            || resources.scratch_bytes > limits.scratch_bytes
            || resources.prepared_bytes > limits.prepared_bytes
        {
            return Err(Error::limit(
                "a pipeline work item exceeds a configured byte budget",
            ));
        }
    }
    Ok(())
}

fn dispatch_work<T>(
    work: Vec<WorkItem<T>>,
    sender: &Sender<QueuedWork<T>>,
    resources: &Arc<ResourceBudget>,
    dispatch_window: &Arc<DispatchWindow>,
    cancellation: &CancellationToken,
    observer: &dyn ExecutionObserver,
) where
    T: Send,
{
    for item in work {
        if cancellation.is_cancelled() {
            break;
        }
        let ordinal = item.ordinal;
        observer.observe(&ExecutionEvent::WorkQueued { ordinal });
        observer.observe(&ExecutionEvent::PhaseStarted {
            phase: ExecutionPhase::QueueWait,
            ordinal: Some(ordinal),
        });
        let queued_at = Instant::now();
        let Ok(window) = dispatch_window.acquire(cancellation) else {
            break;
        };
        let Ok(reservation) = resources.acquire(item.resources, cancellation) else {
            break;
        };
        emit_budget_usage(resources, observer);
        let queued = QueuedWork {
            item,
            queued_at,
            reservation,
            window,
        };
        if sender.send(queued).is_err() {
            break;
        }
    }
}

fn worker_loop<T, U, P>(
    receiver: &Receiver<QueuedWork<T>>,
    sender: &Sender<WorkerOutput<U>>,
    resources: &Arc<ResourceBudget>,
    cancellation: &CancellationToken,
    observer: &dyn ExecutionObserver,
    prepare: &P,
) where
    T: Send,
    U: Send,
    P: Fn(T, &mut PrepareContext<'_>) -> Result<PreparedItem<U>> + Sync,
{
    while let Ok(queued) = receiver.recv() {
        let ordinal = queued.item.ordinal;
        let queue_wait = queued.queued_at.elapsed();
        observer.observe(&ExecutionEvent::PhaseFinished {
            phase: ExecutionPhase::QueueWait,
            ordinal: Some(ordinal),
            duration: queue_wait,
            bytes: 0,
        });

        let mut context = PrepareContext::new(
            ordinal,
            cancellation,
            observer,
            resources,
            queued.reservation,
        );
        observer.observe(&ExecutionEvent::PhaseStarted {
            phase: ExecutionPhase::Preparation,
            ordinal: Some(ordinal),
        });
        let started = Instant::now();
        let result = cancellation
            .check()
            .and_then(|()| prepare(queued.item.value, &mut context));
        let duration = started.elapsed();
        observer.observe(&ExecutionEvent::PhaseFinished {
            phase: ExecutionPhase::Preparation,
            ordinal: Some(ordinal),
            duration,
            bytes: result.as_ref().map_or(0, PreparedItem::bytes),
        });
        context.metrics.push(PhaseMetric {
            phase: ExecutionPhase::Preparation,
            duration,
        });
        if let Ok(prepared) = &result {
            observer.observe(&ExecutionEvent::WorkPrepared {
                ordinal,
                bytes: prepared.bytes,
            });
        }
        let reserved_prepared_bytes = context.reserved.prepared_bytes;
        let result = result.and_then(|prepared| {
            if prepared.bytes > reserved_prepared_bytes {
                Err(Error::limit(
                    "actual prepared bytes exceed the work-item prepared reservations",
                ))
            } else {
                Ok(prepared)
            }
        });
        let actual_prepared_bytes = result.as_ref().map_or(0, PreparedItem::bytes);
        let PrepareContext {
            initial_reservation,
            promoted_prepared,
            mut metrics,
            ..
        } = context;
        let reservation = result.as_ref().ok().map(|_| {
            OutputReservation::new(initial_reservation, promoted_prepared)
                .retain_prepared(actual_prepared_bytes)
        });
        emit_budget_usage(resources, observer);
        metrics.push(PhaseMetric {
            phase: ExecutionPhase::QueueWait,
            duration: queue_wait,
        });

        let output = WorkerOutput {
            ordinal,
            result,
            reservation,
            window: queued.window,
            metrics,
        };
        if sender.send(output).is_err() {
            break;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CoordinatorBounds<'a> {
    resources: &'a ResourceBudget,
    pending_delivery_limit: usize,
}

fn coordinate_results<U, S, O>(
    ordinals: &[u64],
    receiver: &Receiver<WorkerOutput<U>>,
    sink: &mut S,
    observer: &O,
    cancellation: &CancellationToken,
    bounds: CoordinatorBounds<'_>,
    report: &mut ExecutionReportBuilder,
) -> Result<()>
where
    S: PreparedSink<U>,
    O: ExecutionObserver,
{
    let mut pending = BTreeMap::new();
    let mut next_index = 0;
    let mut terminal_error = None;

    while next_index < ordinals.len() {
        let Ok(output) = receiver.recv() else {
            break;
        };
        for metric in &output.metrics {
            report.add_phase(metric.phase, metric.duration);
        }
        if output.result.is_ok() {
            report.prepared();
        } else {
            report.failed();
        }
        if pending.insert(output.ordinal, output).is_some() {
            terminal_error = Some(Error::delivery(
                "a pipeline worker produced a duplicate ordinal",
            ));
            cancellation.cancel();
            break;
        }

        while next_index < ordinals.len() {
            let Some(output) = pending.remove(&ordinals[next_index]) else {
                break;
            };
            match deliver_output(
                output,
                sink,
                observer,
                cancellation,
                bounds.resources,
                report,
            ) {
                Ok(()) => {
                    next_index += 1;
                }
                Err(error) => {
                    terminal_error = Some(error);
                    cancellation.cancel();
                    break;
                }
            }
        }
        report.observe_delivery_queue_depth(pending.len());
        observer.observe(&ExecutionEvent::DeliveryQueueDepth {
            queued: pending.len(),
            limit: bounds.pending_delivery_limit,
        });
        if terminal_error.is_some() {
            break;
        }
    }

    drop(pending);
    if let Some(error) = terminal_error {
        if error.is_cancelled() {
            observer.observe(&ExecutionEvent::Cancelled);
        }
        return Err(error);
    }
    if next_index == ordinals.len() {
        return Ok(());
    }
    cancellation.cancel();
    observer.observe(&ExecutionEvent::Cancelled);
    Err(Error::cancelled())
}

fn deliver_output<U, S>(
    output: WorkerOutput<U>,
    sink: &mut S,
    observer: &dyn ExecutionObserver,
    cancellation: &CancellationToken,
    resources: &ResourceBudget,
    report: &mut ExecutionReportBuilder,
) -> Result<()>
where
    S: PreparedSink<U>,
{
    let WorkerOutput {
        ordinal,
        result,
        reservation,
        window,
        ..
    } = output;
    let prepared = result?;
    let bytes = prepared.bytes;
    observer.observe(&ExecutionEvent::PhaseStarted {
        phase: ExecutionPhase::DeliveryCallback,
        ordinal: Some(ordinal),
    });
    let started = Instant::now();
    let delivered = sink.deliver(ordinal, prepared, cancellation);
    let duration = started.elapsed();
    observer.observe(&ExecutionEvent::PhaseFinished {
        phase: ExecutionPhase::DeliveryCallback,
        ordinal: Some(ordinal),
        duration,
        bytes,
    });
    report.add_phase(ExecutionPhase::DeliveryCallback, duration);
    if delivered.is_err() {
        report.failed();
    }
    delivered?;
    report.delivered(bytes);
    observer.observe(&ExecutionEvent::WorkDelivered { ordinal, bytes });
    drop(reservation);
    drop(window);
    emit_budget_usage(resources, observer);
    Ok(())
}

fn emit_budget_usage(resources: &ResourceBudget, observer: &dyn ExecutionObserver) {
    let snapshot = resources.snapshot_lossy();
    for (kind, used, limit) in [
        (
            MemoryKind::Source,
            snapshot.used.source_bytes,
            snapshot.limit.source_bytes,
        ),
        (
            MemoryKind::Scratch,
            snapshot.used.scratch_bytes,
            snapshot.limit.scratch_bytes,
        ),
        (
            MemoryKind::Prepared,
            snapshot.used.prepared_bytes,
            snapshot.limit.prepared_bytes,
        ),
    ] {
        observer.observe(&ExecutionEvent::BudgetUsage { kind, used, limit });
    }
}

#[derive(Debug)]
struct QueuedWork<T> {
    item: WorkItem<T>,
    queued_at: Instant,
    reservation: ResourcePermit,
    window: DispatchPermit,
}

#[derive(Debug, Clone, Copy)]
struct PhaseMetric {
    phase: ExecutionPhase,
    duration: Duration,
}

#[derive(Debug)]
struct WorkerOutput<T> {
    ordinal: u64,
    result: Result<PreparedItem<T>>,
    reservation: Option<OutputReservation>,
    window: DispatchPermit,
    metrics: Vec<PhaseMetric>,
}

#[derive(Debug, Clone, Copy)]
struct BudgetSnapshot {
    used: ResourceWeights,
    limit: ResourceWeights,
}

#[derive(Debug)]
struct ResourceState {
    used: ResourceWeights,
    peaks: ResourceWeights,
}

#[derive(Debug)]
struct ResourceBudget {
    limit: ResourceWeights,
    state: Mutex<ResourceState>,
    changed: Condvar,
}

impl ResourceBudget {
    const fn new(limits: &ExecutionLimits) -> Self {
        Self {
            limit: ResourceWeights::new(
                limits.source_bytes,
                limits.scratch_bytes,
                limits.prepared_bytes,
            ),
            state: Mutex::new(ResourceState {
                used: ResourceWeights::new(0, 0, 0),
                peaks: ResourceWeights::new(0, 0, 0),
            }),
            changed: Condvar::new(),
        }
    }

    fn acquire(
        self: &Arc<Self>,
        requested: ResourceWeights,
        cancellation: &CancellationToken,
    ) -> Result<ResourcePermit> {
        cancellation.check()?;
        if !fits(ResourceWeights::default(), requested, self.limit) {
            return Err(Error::limit(
                "a deferred pipeline resource request exceeds a configured byte budget",
            ));
        }
        loop {
            cancellation.check()?;
            let state = self.state.lock().map_err(|_poisoned| {
                Error::delivery("pipeline resource budget lock was poisoned")
            })?;
            if fits(state.used, requested, self.limit) {
                let mut state = state;
                state.used = add_weights(state.used, requested);
                state.peaks = max_weights(state.peaks, state.used);
                return Ok(ResourcePermit {
                    budget: Arc::clone(self),
                    reserved: requested,
                });
            }
            let wait = self
                .changed
                .wait_timeout(state, CANCELLATION_POLL_INTERVAL)
                .map_err(|_poisoned| {
                    Error::delivery("pipeline resource budget lock was poisoned")
                })?;
            drop(wait);
        }
    }

    fn peaks(&self) -> Result<PeakBytes> {
        let state = self
            .state
            .lock()
            .map_err(|_poisoned| Error::delivery("pipeline resource budget lock was poisoned"))?;
        Ok(PeakBytes::new(
            state.peaks.source_bytes,
            state.peaks.scratch_bytes,
            state.peaks.prepared_bytes,
        ))
    }

    fn snapshot_lossy(&self) -> BudgetSnapshot {
        let state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        BudgetSnapshot {
            used: state.used,
            limit: self.limit,
        }
    }

    fn release(&self, released: ResourceWeights) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.used = subtract_weights(state.used, released);
        drop(state);
        self.changed.notify_all();
    }
}

#[derive(Debug)]
struct ResourcePermit {
    budget: Arc<ResourceBudget>,
    reserved: ResourceWeights,
}

impl ResourcePermit {
    fn release_prepared(&mut self) {
        let released = ResourceWeights::new(0, 0, self.reserved.prepared_bytes);
        self.budget.release(released);
        self.reserved.prepared_bytes = 0;
    }

    fn split_prepared(mut self) -> (Self, Option<Self>) {
        let prepared_bytes = self.reserved.prepared_bytes;
        self.reserved.prepared_bytes = 0;
        let prepared = (prepared_bytes > 0).then(|| Self {
            budget: Arc::clone(&self.budget),
            reserved: ResourceWeights::new(0, 0, prepared_bytes),
        });
        (self, prepared)
    }

    fn retain_prepared(mut self, prepared_bytes: u64) -> Self {
        let retained_bytes = self.reserved.prepared_bytes.min(prepared_bytes);
        let retained = ResourceWeights::new(0, 0, retained_bytes);
        let released = subtract_weights(self.reserved, retained);
        self.budget.release(released);
        self.reserved = retained;
        self
    }
}

impl Drop for ResourcePermit {
    fn drop(&mut self) {
        self.budget.release(self.reserved);
    }
}

#[derive(Debug)]
struct OutputReservation {
    initial: ResourcePermit,
    promoted_prepared: Vec<ResourcePermit>,
}

impl OutputReservation {
    const fn new(initial: ResourcePermit, promoted_prepared: Vec<ResourcePermit>) -> Self {
        Self {
            initial,
            promoted_prepared,
        }
    }

    fn retain_prepared(mut self, prepared_bytes: u64) -> Self {
        let mut remaining = prepared_bytes;
        self.initial = self.initial.retain_prepared(remaining);
        remaining = remaining.saturating_sub(self.initial.reserved.prepared_bytes);
        for reservation in &mut self.promoted_prepared {
            let retained = reservation.reserved.prepared_bytes.min(remaining);
            let released =
                subtract_weights(reservation.reserved, ResourceWeights::new(0, 0, retained));
            reservation.budget.release(released);
            reservation.reserved = ResourceWeights::new(0, 0, retained);
            remaining -= retained;
        }
        debug_assert_eq!(
            remaining, 0,
            "prepared-byte validation guarantees sufficient reservations"
        );
        self
    }
}

#[derive(Debug)]
struct DispatchState {
    used: usize,
}

#[derive(Debug)]
struct DispatchWindow {
    limit: usize,
    state: Mutex<DispatchState>,
    changed: Condvar,
}

impl DispatchWindow {
    const fn new(limit: usize) -> Self {
        Self {
            limit,
            state: Mutex::new(DispatchState { used: 0 }),
            changed: Condvar::new(),
        }
    }

    fn acquire(self: &Arc<Self>, cancellation: &CancellationToken) -> Result<DispatchPermit> {
        loop {
            cancellation.check()?;
            let state = self.state.lock().map_err(|_poisoned| {
                Error::delivery("pipeline dispatch window lock was poisoned")
            })?;
            if state.used < self.limit {
                let mut state = state;
                state.used += 1;
                return Ok(DispatchPermit {
                    window: Arc::clone(self),
                });
            }
            let wait = self
                .changed
                .wait_timeout(state, CANCELLATION_POLL_INTERVAL)
                .map_err(|_poisoned| {
                    Error::delivery("pipeline dispatch window lock was poisoned")
                })?;
            drop(wait);
        }
    }

    fn release(&self) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.used = state.used.saturating_sub(1);
        drop(state);
        self.changed.notify_all();
    }
}

#[derive(Debug)]
struct DispatchPermit {
    window: Arc<DispatchWindow>,
}

impl Drop for DispatchPermit {
    fn drop(&mut self) {
        self.window.release();
    }
}

const fn fits(used: ResourceWeights, requested: ResourceWeights, limit: ResourceWeights) -> bool {
    requested.source_bytes <= limit.source_bytes - used.source_bytes
        && requested.scratch_bytes <= limit.scratch_bytes - used.scratch_bytes
        && requested.prepared_bytes <= limit.prepared_bytes - used.prepared_bytes
}

const fn add_weights(left: ResourceWeights, right: ResourceWeights) -> ResourceWeights {
    ResourceWeights::new(
        left.source_bytes + right.source_bytes,
        left.scratch_bytes + right.scratch_bytes,
        left.prepared_bytes + right.prepared_bytes,
    )
}

const fn subtract_weights(left: ResourceWeights, right: ResourceWeights) -> ResourceWeights {
    ResourceWeights::new(
        left.source_bytes.saturating_sub(right.source_bytes),
        left.scratch_bytes.saturating_sub(right.scratch_bytes),
        left.prepared_bytes.saturating_sub(right.prepared_bytes),
    )
}

const fn max_weights(left: ResourceWeights, right: ResourceWeights) -> ResourceWeights {
    ResourceWeights::new(
        if left.source_bytes > right.source_bytes {
            left.source_bytes
        } else {
            right.source_bytes
        },
        if left.scratch_bytes > right.scratch_bytes {
            left.scratch_bytes
        } else {
            right.scratch_bytes
        },
        if left.prepared_bytes > right.prepared_bytes {
            left.prepared_bytes
        } else {
            right.prepared_bytes
        },
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{Pipeline, PreparedItem, ResourceWeights, WorkItem};
    use crate::limits::ExecutionLimits;
    use crate::telemetry::{ExecutionEvent, MemoryKind};
    use crate::{CancellationToken, Result};

    #[test]
    fn deferred_prepared_promotion_remains_reserved_through_delivery() -> Result<()> {
        let pipeline = Pipeline::new(ExecutionLimits {
            workers: 1,
            max_work_items: 1,
            delivery_queue_depth: 1,
            dispatch_lookahead: 2,
            source_bytes: 1,
            scratch_bytes: 1,
            prepared_bytes: 8,
        })?;
        let latest_prepared = Arc::new(AtomicU64::new(0));
        let observer_prepared = Arc::clone(&latest_prepared);
        let observer = move |event: &ExecutionEvent| {
            if let ExecutionEvent::BudgetUsage {
                kind: MemoryKind::Prepared,
                used,
                ..
            } = event
            {
                observer_prepared.store(*used, Ordering::SeqCst);
            }
        };
        let mut prepared_during_delivery = 0_u64;
        let report = pipeline.execute(
            [WorkItem::new(0, (), ResourceWeights::new(0, 0, 2))],
            |(), context| {
                context.with_resources(ResourceWeights::new(0, 0, 6), |_context| {
                    Ok(PreparedItem::new((), 6))
                })
            },
            &mut |_ordinal, _item: PreparedItem<()>, cancellation: &CancellationToken| {
                cancellation.check()?;
                prepared_during_delivery = latest_prepared.load(Ordering::SeqCst);
                Ok(())
            },
            &observer,
        )?;

        assert_eq!(prepared_during_delivery, 6);
        assert_eq!(report.peak_bytes().prepared(), 6);
        Ok(())
    }
}
