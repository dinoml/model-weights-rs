//! Focused pipeline ordering, cancellation, telemetry, and bound tests.

use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{TryRecvError, bounded, unbounded};
use model_weights::identity::{ContentDigest, StableName};
use model_weights::limits::ExecutionLimits;
use model_weights::pipeline::{Pipeline, PrepareContext, PreparedItem, ResourceWeights, WorkItem};
use model_weights::telemetry::{ExecutionEvent, ExecutionPhase, NoopObserver};
use model_weights::{CancellationToken, Result};

fn limits(workers: usize) -> ExecutionLimits {
    ExecutionLimits {
        workers,
        max_work_items: 100,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 100,
        scratch_bytes: 100,
        prepared_bytes: 100,
    }
}

fn work(count: u64, resources: ResourceWeights) -> Vec<WorkItem<u64>> {
    (0..count)
        .map(|ordinal| WorkItem::new(ordinal, ordinal, resources))
        .collect()
}

#[test]
fn parallel_completion_is_delivered_in_ordinal_order() -> Result<()> {
    let pipeline = Pipeline::new(limits(3))?;
    let (later_ready_sender, later_ready_receiver) = bounded(1);
    let (first_release_sender, first_release_receiver) = bounded(1);
    let handle = thread::spawn(move || {
        let mut delivered = Vec::new();
        let report = pipeline.execute(
            work(3, ResourceWeights::new(1, 1, 1)),
            move |value, _context: &mut PrepareContext<'_>| {
                if value == 0 {
                    first_release_receiver
                        .recv()
                        .expect("test release sender must remain connected");
                } else if value == 1 {
                    later_ready_sender
                        .send(())
                        .expect("test readiness receiver must remain connected");
                }
                Ok(PreparedItem::new(value, 1))
            },
            &mut |ordinal, item: PreparedItem<u64>, _cancellation: &CancellationToken| {
                delivered.push((ordinal, item.into_value()));
                Ok(())
            },
            &NoopObserver,
        )?;
        Ok::<_, model_weights::Error>((delivered, report))
    });

    later_ready_receiver
        .recv()
        .expect("later work item must complete first");
    first_release_sender
        .send(())
        .expect("pipeline worker must remain connected");
    let (delivered, report) = handle.join().expect("pipeline thread must not panic")?;

    assert_eq!(delivered, vec![(0, 0), (1, 1), (2, 2)]);
    assert_eq!(report.counters().delivered(), 3);
    Ok(())
}

#[test]
fn work_item_limit_is_enforced_before_workers_start() -> Result<()> {
    let mut limits = limits(1);
    limits.max_work_items = 1;
    let pipeline = Pipeline::new(limits)?;
    let prepared = AtomicBool::new(false);
    let work = [
        WorkItem::new(0, 0_u64, ResourceWeights::default()),
        WorkItem::new(1, 1_u64, ResourceWeights::default()),
    ];

    let error = pipeline
        .execute(
            work,
            |value, _context| {
                prepared.store(true, Ordering::SeqCst);
                Ok(PreparedItem::new(value, 0))
            },
            &mut |_ordinal, _item: PreparedItem<u64>, _cancellation: &CancellationToken| Ok(()),
            &NoopObserver,
        )
        .expect_err("work metadata beyond the configured limit was accepted");

    assert_eq!(
        error.category(),
        model_weights::ErrorCategory::ResourceLimit
    );
    assert!(!prepared.load(Ordering::SeqCst));
    Ok(())
}

#[test]
fn dispatch_lookahead_bounds_work_independently_of_result_queue_depth() -> Result<()> {
    let queue_limited = ExecutionLimits {
        workers: 8,
        max_work_items: 32,
        delivery_queue_depth: 1,
        dispatch_lookahead: 3,
        source_bytes: 1,
        scratch_bytes: 1,
        prepared_bytes: 1,
    };
    let pipeline = Pipeline::new(queue_limited)?;
    let (first_started_sender, first_started_receiver) = bounded(1);
    let (first_release_sender, first_release_receiver) = bounded(1);
    let (later_finished_sender, later_finished_receiver) = unbounded();
    let handle = thread::spawn(move || {
        pipeline.execute(
            work(32, ResourceWeights::default()),
            move |value, _context: &mut PrepareContext<'_>| {
                if value == 0 {
                    first_started_sender
                        .send(())
                        .expect("test readiness receiver must remain connected");
                    first_release_receiver
                        .recv()
                        .expect("test release sender must remain connected");
                } else {
                    later_finished_sender
                        .send(value)
                        .expect("test completion receiver must remain connected");
                }
                Ok(PreparedItem::new(value, 0))
            },
            &mut |_ordinal, _item: PreparedItem<u64>, _cancellation: &CancellationToken| Ok(()),
            &NoopObserver,
        )
    });

    first_started_receiver
        .recv()
        .expect("first ordinal must be in flight");
    later_finished_receiver
        .recv()
        .expect("first later item must finish");
    later_finished_receiver
        .recv()
        .expect("second later item must finish");
    assert_eq!(later_finished_receiver.try_recv(), Err(TryRecvError::Empty));
    first_release_sender
        .send(())
        .expect("first worker must remain connected");
    let report = handle.join().expect("pipeline thread must not panic")?;

    assert_eq!(report.peak_delivery_queue_depth(), 2);
    assert_eq!(report.counters().delivered(), 32);
    Ok(())
}

#[test]
fn zero_dispatch_lookahead_is_rejected() {
    let mut invalid = limits(1);
    invalid.dispatch_lookahead = 0;

    let error = Pipeline::new(invalid).expect_err("zero dispatch lookahead was accepted");

    assert_eq!(
        error.category(),
        model_weights::ErrorCategory::ResourceLimit
    );
}

#[test]
fn default_worker_count_uses_available_parallelism() {
    let expected = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let limits = ExecutionLimits::default();

    assert_eq!(limits.workers, expected);
    assert_eq!(limits.dispatch_lookahead, expected);
}

#[test]
fn preparation_errors_are_reported_by_lowest_ordinal() -> Result<()> {
    let pipeline = Pipeline::new(limits(2))?;
    let (high_failed_sender, high_failed_receiver) = bounded(1);
    let (low_release_sender, low_release_receiver) = bounded(1);
    let handle = thread::spawn(move || {
        pipeline.execute(
            work(2, ResourceWeights::new(1, 1, 1)),
            move |value, _context: &mut PrepareContext<'_>| {
                if value == 0 {
                    low_release_receiver
                        .recv()
                        .expect("test release sender must remain connected");
                    StableName::parse("")?;
                } else {
                    high_failed_sender
                        .send(())
                        .expect("test readiness receiver must remain connected");
                    let _digest = "not-a-digest".parse::<ContentDigest>()?;
                }
                Ok(PreparedItem::new(value, 1))
            },
            &mut |_ordinal, _item: PreparedItem<u64>, _cancellation: &CancellationToken| Ok(()),
            &NoopObserver,
        )
    });

    high_failed_receiver
        .recv()
        .expect("higher ordinal must fail before release");
    low_release_sender
        .send(())
        .expect("pipeline worker must remain connected");
    let error = handle
        .join()
        .expect("pipeline thread must not panic")
        .expect_err("pipeline must return a preparation error");

    assert!(error.message().contains("stable name"));
    Ok(())
}

#[test]
fn worker_count_never_exceeds_configured_limit() -> Result<()> {
    let pipeline = Pipeline::new(limits(2))?;
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let (started_sender, started_receiver) = unbounded();
    let (release_sender, release_receiver) = unbounded();
    let worker_active = Arc::clone(&active);
    let worker_peak = Arc::clone(&peak);
    let handle = thread::spawn(move || {
        pipeline.execute(
            work(6, ResourceWeights::new(1, 1, 1)),
            move |value, _context: &mut PrepareContext<'_>| {
                let now = worker_active.fetch_add(1, Ordering::SeqCst) + 1;
                worker_peak.fetch_max(now, Ordering::SeqCst);
                started_sender
                    .send(value)
                    .expect("test readiness receiver must remain connected");
                release_receiver
                    .recv()
                    .expect("test release sender must remain connected");
                worker_active.fetch_sub(1, Ordering::SeqCst);
                Ok(PreparedItem::new(value, 1))
            },
            &mut |_ordinal, _item: PreparedItem<u64>, _cancellation: &CancellationToken| Ok(()),
            &NoopObserver,
        )
    });

    started_receiver.recv().expect("first worker must start");
    started_receiver.recv().expect("second worker must start");
    assert_eq!(started_receiver.try_recv(), Err(TryRecvError::Empty));
    for _ in 0..6 {
        release_sender
            .send(())
            .expect("pipeline workers must remain connected");
    }
    let report = handle.join().expect("pipeline thread must not panic")?;

    assert_eq!(peak.load(Ordering::SeqCst), 2);
    assert_eq!(report.counters().delivered(), 6);
    Ok(())
}

#[test]
fn weighted_memory_budget_applies_backpressure_and_reports_peaks() -> Result<()> {
    let constrained = ExecutionLimits {
        workers: 3,
        max_work_items: 3,
        delivery_queue_depth: 2,
        dispatch_lookahead: 3,
        source_bytes: 2,
        scratch_bytes: 3,
        prepared_bytes: 5,
    };
    let pipeline = Pipeline::new(constrained)?;
    let (started_sender, started_receiver) = unbounded();
    let (release_sender, release_receiver) = unbounded();
    let handle = thread::spawn(move || {
        pipeline.execute(
            work(3, ResourceWeights::new(2, 3, 5)),
            move |value, _context: &mut PrepareContext<'_>| {
                started_sender
                    .send(value)
                    .expect("test readiness receiver must remain connected");
                release_receiver
                    .recv()
                    .expect("test release sender must remain connected");
                Ok(PreparedItem::new(value, 5))
            },
            &mut |_ordinal, _item: PreparedItem<u64>, _cancellation: &CancellationToken| Ok(()),
            &NoopObserver,
        )
    });

    started_receiver
        .recv()
        .expect("first reserved work item must start");
    assert_eq!(started_receiver.try_recv(), Err(TryRecvError::Empty));
    for index in 0..3 {
        release_sender
            .send(())
            .expect("pipeline workers must remain connected");
        if index < 2 {
            started_receiver
                .recv()
                .expect("next reserved work item must start");
        }
    }
    let report = handle.join().expect("pipeline thread must not panic")?;

    assert_eq!(report.peak_bytes().source(), 2);
    assert_eq!(report.peak_bytes().scratch(), 3);
    assert_eq!(report.peak_bytes().prepared(), 5);
    Ok(())
}

#[test]
fn cancellation_stops_cooperative_work_promptly() -> Result<()> {
    let cancellation = CancellationToken::new();
    let pipeline = Pipeline::with_cancellation(limits(1), cancellation.clone())?;
    let (started_sender, started_receiver) = bounded(1);
    let handle = thread::spawn(move || {
        pipeline.execute(
            work(1, ResourceWeights::new(10, 10, 10)),
            move |_value, context: &mut PrepareContext<'_>| {
                started_sender
                    .send(())
                    .expect("test readiness receiver must remain connected");
                loop {
                    context.cancellation().check()?;
                    thread::yield_now();
                }
            },
            &mut |_ordinal, _item: PreparedItem<u64>, _cancellation: &CancellationToken| Ok(()),
            &NoopObserver,
        )
    });

    started_receiver
        .recv()
        .expect("cooperative work item must start");
    cancellation.cancel();
    let error = handle
        .join()
        .expect("pipeline thread must not panic")
        .expect_err("cancelled pipeline must return an error");

    assert!(error.is_cancelled());
    Ok(())
}

#[test]
fn cancellation_drops_prepared_items_waiting_for_earlier_ordinals() -> Result<()> {
    let cancellation = CancellationToken::new();
    let pipeline = Pipeline::with_cancellation(limits(2), cancellation.clone())?;
    let dropped = Arc::new(AtomicUsize::new(0));
    let sink_calls = Arc::new(AtomicUsize::new(0));
    let (first_started_sender, first_started_receiver) = bounded(1);
    let (second_prepared_sender, second_prepared_receiver) = bounded(1);
    let worker_dropped = Arc::clone(&dropped);
    let worker_sink_calls = Arc::clone(&sink_calls);
    let handle = thread::spawn(move || {
        pipeline.execute(
            work(2, ResourceWeights::new(1, 1, 10)),
            move |value, context: &mut PrepareContext<'_>| {
                if value == 0 {
                    first_started_sender
                        .send(())
                        .expect("test readiness receiver must remain connected");
                    loop {
                        context.cancellation().check()?;
                        thread::yield_now();
                    }
                }
                second_prepared_sender
                    .send(())
                    .expect("test readiness receiver must remain connected");
                Ok(PreparedItem::new(
                    DropProbe(Arc::clone(&worker_dropped)),
                    10,
                ))
            },
            &mut move |_ordinal,
                       _item: PreparedItem<DropProbe>,
                       _cancellation: &CancellationToken| {
                worker_sink_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            &NoopObserver,
        )
    });

    first_started_receiver
        .recv()
        .expect("first ordinal must be in flight");
    second_prepared_receiver
        .recv()
        .expect("second ordinal must prepare out of order");
    cancellation.cancel();
    let error = handle
        .join()
        .expect("pipeline thread must not panic")
        .expect_err("cancelled pipeline must return an error");

    assert!(error.is_cancelled());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(sink_calls.load(Ordering::SeqCst), 0);
    Ok(())
}

struct DropProbe(Arc<AtomicUsize>);

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn observer_and_report_expose_typed_phase_and_budget_measurements()
-> std::result::Result<(), Box<dyn StdError>> {
    let pipeline = Pipeline::new(limits(1))?;
    let events = Arc::new(Mutex::new(Vec::new()));
    let event_store = Arc::clone(&events);
    let observer = move |event: &ExecutionEvent| {
        event_store
            .lock()
            .expect("test event mutex must not be poisoned")
            .push(event.clone());
    };
    let mut delivered = None;
    let report = pipeline.execute(
        work(1, ResourceWeights::new(3, 4, 5)),
        |value, context: &mut PrepareContext<'_>| {
            context.measure(ExecutionPhase::Transform, 5, |_cancellation| ());
            Ok(PreparedItem::new(value, 5))
        },
        &mut |_ordinal, item: PreparedItem<u64>, _cancellation: &CancellationToken| {
            delivered = Some(item.into_value());
            Ok(())
        },
        &observer,
    )?;
    let events = events
        .lock()
        .expect("test event mutex must not be poisoned");

    assert_eq!(delivered, Some(0));
    assert!(
        report
            .phase_durations()
            .any(|(phase, _duration)| phase == ExecutionPhase::Transform)
    );
    assert!(events.iter().any(|event| matches!(
        event,
        ExecutionEvent::PhaseFinished {
            phase: ExecutionPhase::Transform,
            ..
        }
    )));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ExecutionEvent::BudgetUsage { .. }))
    );
    Ok(())
}
