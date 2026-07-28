//! Focused cache integrity, recovery, concurrency, and eviction tests.

use std::error::Error as StdError;
use std::fs;
use std::io::{self, Read};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded};
use model_weights::cache::{
    Cache, CacheCompatibility, CacheKey, CacheLookup, CacheMissReason, CacheNamespace,
    CacheOptions, CachePublication, CacheValidation, EvictionReason,
};
use model_weights::identity::ContentDigest;
use model_weights::{CancellationToken, ErrorCategory};
use tempfile::TempDir;

fn prepared_compatibility() -> CacheCompatibility {
    CacheCompatibility::prepared(
        7,
        ContentDigest::from_bytes([1; 32]),
        ContentDigest::from_bytes([2; 32]),
    )
}

fn key_for(source: &[u8]) -> CacheKey {
    CacheKey::derive(
        CacheNamespace::Prepared,
        &prepared_compatibility(),
        [source, b"plan"],
    )
}

#[test]
fn cache_startup_and_maintenance_scans_are_bounded_and_cancellable() -> Result<(), Box<dyn StdError>>
{
    let temporary = TempDir::new()?;
    let cancelled_root = temporary.path().join("cancelled");
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let cancelled = Cache::open_with_options_and_cancellation(
        &cancelled_root,
        CacheOptions::default(),
        &cancellation,
    )
    .expect_err("pre-cancelled startup must fail");
    assert_eq!(cancelled.category(), ErrorCategory::Cancelled);
    assert!(!cancelled_root.exists());

    let bounded_root = temporary.path().join("bounded");
    let cache = Cache::open_with_options(
        &bounded_root,
        CacheOptions::default().maximum_scan_entries(1),
    )?;
    let bounded_key = key_for(b"bounded-scan-source");
    let bounded_entry = cache.entry_path(CacheNamespace::Prepared, bounded_key);
    fs::create_dir_all(&bounded_entry)?;
    let exceeded = cache
        .inspect()
        .expect_err("the configured scan-entry bound must be enforced");
    assert_eq!(exceeded.category(), ErrorCategory::ResourceLimit);

    let shard = bounded_entry
        .parent()
        .expect("a cache entry must have a shard parent");
    fs::create_dir(shard.join("unrelated-sibling"))?;
    let exceeded = cache
        .lookup(
            CacheNamespace::Prepared,
            bounded_key,
            &prepared_compatibility(),
        )
        .expect_err("replacement discovery must share the configured scan bound");
    assert_eq!(exceeded.category(), ErrorCategory::ResourceLimit);

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let cancelled = cache
        .recover_stale_with_cancellation(&cancellation)
        .expect_err("pre-cancelled recovery must fail");
    assert_eq!(cancelled.category(), ErrorCategory::Cancelled);
    Ok(())
}

#[test]
fn zero_cache_scan_limit_is_rejected_before_creating_storage() -> Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let root = temporary.path().join("invalid");
    let error = Cache::open_with_options(&root, CacheOptions::default().maximum_scan_entries(0))
        .expect_err("a zero scan-entry bound must be rejected");

    assert_eq!(error.category(), ErrorCategory::ResourceLimit);
    assert!(!root.exists());
    Ok(())
}

#[test]
fn cache_key_is_stable_and_changes_with_source_identity() {
    let first = key_for(b"source-a");
    let repeated = key_for(b"source-a");
    let changed = key_for(b"source-b");

    assert_eq!(
        first.to_string(),
        "e7b1323bb436bae097263b6f73f33dbd1fde4b605585e01cfd10258cef2bd468"
    );
    assert_eq!(first, repeated);
    assert_ne!(first, changed);
}

#[test]
fn lookup_rejects_same_length_payload_corruption() -> Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let cache = Cache::open(temporary.path())?;
    let compatibility = prepared_compatibility();
    let key = key_for(b"corruption-source");
    let publication = cache.publish_bytes(
        CacheNamespace::Prepared,
        key,
        &compatibility,
        b"hello-cache",
    )?;
    let payload_path = publication.entry().info().payload_path().to_path_buf();
    drop(publication);

    fs::write(payload_path, b"jello-cache")?;
    let lookup = cache.lookup(CacheNamespace::Prepared, key, &compatibility)?;

    assert!(matches!(
        lookup,
        CacheLookup::Miss(CacheMissReason::DigestMismatch { .. })
    ));
    Ok(())
}

#[test]
fn trusted_metadata_validation_explicitly_skips_warm_payload_rehash()
-> Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let cache = Cache::open(temporary.path())?;
    let compatibility = prepared_compatibility();
    let key = key_for(b"trusted-metadata-source");
    let publication = cache.publish_bytes(
        CacheNamespace::Prepared,
        key,
        &compatibility,
        b"hello-cache",
    )?;
    let payload_path = publication.entry().info().payload_path().to_path_buf();
    drop(publication);
    fs::write(payload_path, b"jello-cache")?;

    let trusted = cache.lookup_with_validation(
        CacheNamespace::Prepared,
        key,
        &compatibility,
        CacheValidation::TrustedMetadata,
    )?;
    let full = cache.lookup(CacheNamespace::Prepared, key, &compatibility)?;

    assert!(matches!(trusted, CacheLookup::Hit(_)));
    assert!(matches!(
        full,
        CacheLookup::Miss(CacheMissReason::DigestMismatch { .. })
    ));
    Ok(())
}

#[test]
fn lookup_rejects_transform_and_backend_abi_changes() -> Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let cache = Cache::open(temporary.path())?;
    let compatibility = prepared_compatibility();
    let key = key_for(b"compatibility-source");
    cache.publish_bytes(CacheNamespace::Prepared, key, &compatibility, b"prepared")?;
    let changed = CacheCompatibility::prepared(
        compatibility.format_version(),
        ContentDigest::from_bytes([3; 32]),
        ContentDigest::from_bytes([4; 32]),
    );

    let lookup = cache.lookup(CacheNamespace::Prepared, key, &changed)?;

    assert!(matches!(
        lookup,
        CacheLookup::Miss(CacheMissReason::CompatibilityMismatch { .. })
    ));
    Ok(())
}

#[test]
fn publication_replaces_a_corrupt_entry_under_the_writer_lease() -> Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let cache = Cache::open(temporary.path())?;
    let compatibility = prepared_compatibility();
    let key = key_for(b"replacement-source");
    let first = cache.publish_bytes(
        CacheNamespace::Prepared,
        key,
        &compatibility,
        b"first-value",
    )?;
    let payload_path = first.entry().info().payload_path().to_path_buf();
    drop(first);
    fs::write(payload_path, b"wrong-value")?;
    assert!(matches!(
        cache.lookup(CacheNamespace::Prepared, key, &compatibility)?,
        CacheLookup::Miss(CacheMissReason::DigestMismatch { .. })
    ));

    let replacement = cache.publish_bytes(
        CacheNamespace::Prepared,
        key,
        &compatibility,
        b"final-value",
    )?;
    let mut bytes = Vec::new();
    replacement
        .into_entry()
        .into_payload()
        .read_to_end(&mut bytes)?;

    assert_eq!(bytes, b"final-value");
    assert!(matches!(
        cache.lookup(CacheNamespace::Prepared, key, &compatibility)?,
        CacheLookup::Hit(_)
    ));
    Ok(())
}

#[test]
fn duplicate_writers_publish_exactly_one_complete_entry() -> Result<(), Box<dyn StdError>> {
    const WRITERS: usize = 8;

    let temporary = TempDir::new()?;
    let cache = Cache::open(temporary.path())?;
    let compatibility = prepared_compatibility();
    let key = key_for(b"duplicate-source");
    let barrier = Arc::new(Barrier::new(WRITERS));
    let mut handles = Vec::new();

    for _ in 0..WRITERS {
        let cache = cache.clone();
        let compatibility = compatibility.clone();
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            cache.publish_bytes(
                CacheNamespace::Prepared,
                key,
                &compatibility,
                b"identical-payload",
            )
        }));
    }

    let mut published = 0;
    let mut reused = 0;
    for handle in handles {
        match handle.join().expect("cache writer thread must not panic")? {
            CachePublication::Published(_) => published += 1,
            CachePublication::Reused(_) => reused += 1,
            _ => panic!("cache publication returned an unknown outcome"),
        }
    }
    assert_eq!(published, 1);
    assert_eq!(reused, WRITERS - 1);

    let lookup = cache.lookup(CacheNamespace::Prepared, key, &compatibility)?;
    assert!(matches!(lookup, CacheLookup::Hit(_)));
    Ok(())
}

#[test]
fn interrupted_staging_is_recovered_without_becoming_visible() -> Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let options = CacheOptions::default().stale_after(Duration::ZERO);
    let cache = Cache::open_with_options(temporary.path(), options)?;
    let key = key_for(b"interrupted-source");
    let final_path = cache.entry_path(CacheNamespace::Prepared, key);
    let parent = final_path
        .parent()
        .expect("cache entry path must have a parent");
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(".{key}.tmp-interrupted"));
    fs::create_dir(&staging)?;
    fs::write(staging.join("payload.bin"), b"partial")?;

    let report = cache.recover_stale()?;
    let lookup = cache.lookup(CacheNamespace::Prepared, key, &prepared_compatibility())?;

    assert_eq!(report.removed_directories(), 1);
    assert!(!staging.exists());
    assert!(matches!(
        lookup,
        CacheLookup::Miss(CacheMissReason::NotFound)
    ));
    Ok(())
}

#[test]
fn interrupted_replacement_stays_readable_and_recovery_restores_it() -> Result<(), Box<dyn StdError>>
{
    const READERS: usize = 8;

    let temporary = TempDir::new()?;
    let options = CacheOptions::default().stale_after(Duration::ZERO);
    let cache = Cache::open_with_options(temporary.path(), options)?;
    let compatibility = prepared_compatibility();
    let key = key_for(b"interrupted-replacement-source");
    let publication = cache.publish_bytes(
        CacheNamespace::Prepared,
        key,
        &compatibility,
        b"previous-generation",
    )?;
    drop(publication);

    let final_path = cache.entry_path(CacheNamespace::Prepared, key);
    let parent = final_path
        .parent()
        .expect("cache entry path must have a parent");
    let previous_path = parent.join(format!(".{key}.previous-interrupted"));
    fs::rename(&final_path, &previous_path)?;

    let cache = Arc::new(cache);
    let mut readers = Vec::new();
    for _ in 0..READERS {
        let cache = Arc::clone(&cache);
        let compatibility = compatibility.clone();
        readers.push(thread::spawn(
            move || -> Result<Vec<u8>, model_weights::Error> {
                let entry = cache
                    .lookup(CacheNamespace::Prepared, key, &compatibility)?
                    .into_result()
                    .map_err(|reason| {
                        model_weights::Error::from_category(
                            model_weights::ErrorCategory::Cache,
                            format!("replacement lookup missed: {reason}"),
                        )
                    })?;
                let mut bytes = Vec::new();
                entry
                    .into_payload()
                    .read_to_end(&mut bytes)
                    .map_err(|error| {
                        model_weights::Error::from_category_with_source(
                            model_weights::ErrorCategory::Io,
                            "failed to read replacement payload",
                            error,
                        )
                    })?;
                Ok(bytes)
            },
        ));
    }
    for reader in readers {
        assert_eq!(
            reader.join().expect("cache reader thread must not panic")?,
            b"previous-generation"
        );
    }

    let report = cache.recover_stale()?;
    assert_eq!(report.restored_entries(), 1);
    assert_eq!(report.removed_directories(), 0);
    assert!(final_path.is_dir());
    assert!(!previous_path.exists());
    assert!(matches!(
        cache.lookup(CacheNamespace::Prepared, key, &compatibility)?,
        CacheLookup::Hit(_)
    ));
    Ok(())
}

#[test]
fn interrupted_eviction_is_reclaimed_without_resurrecting_the_entry()
-> Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let options = CacheOptions::default().stale_after(Duration::ZERO);
    let cache = Cache::open_with_options(temporary.path(), options)?;
    let compatibility = prepared_compatibility();
    let key = key_for(b"interrupted-eviction-source");
    let publication = cache.publish_bytes(
        CacheNamespace::Prepared,
        key,
        &compatibility,
        b"evicted-generation",
    )?;
    drop(publication);

    let final_path = cache.entry_path(CacheNamespace::Prepared, key);
    let parent = final_path
        .parent()
        .expect("cache entry path must have a parent");
    let evicted_path = parent.join(format!(".{key}.evicted-interrupted"));
    fs::rename(&final_path, &evicted_path)?;

    let report = cache.recover_stale()?;

    assert_eq!(report.restored_entries(), 0);
    assert_eq!(report.removed_directories(), 1);
    assert!(!final_path.exists());
    assert!(!evicted_path.exists());
    assert!(matches!(
        cache.lookup(CacheNamespace::Prepared, key, &compatibility)?,
        CacheLookup::Miss(CacheMissReason::NotFound)
    ));
    Ok(())
}

#[test]
fn capacity_eviction_accounts_for_and_removes_published_entries() -> Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let cache = Cache::open(temporary.path())?;
    let compatibility = prepared_compatibility();
    for source in [b"one".as_slice(), b"two".as_slice(), b"three".as_slice()] {
        cache.publish_bytes(
            CacheNamespace::Prepared,
            key_for(source),
            &compatibility,
            &[9; 128],
        )?;
    }

    let before = cache.inspect()?;
    let report = cache.evict_to(0, EvictionReason::Capacity, &CancellationToken::new())?;
    let after = cache.inspect()?;

    assert_eq!(before.entries().len(), 3);
    assert_eq!(report.records().len(), 3);
    assert_eq!(after.entries().len(), 0);
    assert_eq!(after.total_bytes(), 0);
    Ok(())
}

#[test]
fn readers_observe_miss_until_atomic_publication_finishes() -> Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let cache = Cache::open(temporary.path())?;
    let compatibility = prepared_compatibility();
    let key = key_for(b"slow-source");
    let (started_sender, started_receiver) = bounded(1);
    let (release_sender, release_receiver) = bounded(1);
    let writer_cache = cache.clone();
    let writer_compatibility = compatibility.clone();

    let writer = thread::spawn(move || {
        writer_cache.publish_reader(
            CacheNamespace::Prepared,
            key,
            &writer_compatibility,
            BlockingReader {
                bytes: Some(b"complete-payload"),
                started: started_sender,
                release: release_receiver,
            },
            &CancellationToken::new(),
        )
    });

    started_receiver.recv()?;
    let during = cache.lookup(CacheNamespace::Prepared, key, &compatibility)?;
    release_sender.send(())?;
    let publication = writer.join().expect("cache writer thread must not panic")?;
    let after = cache.lookup(CacheNamespace::Prepared, key, &compatibility)?;

    assert!(matches!(
        during,
        CacheLookup::Miss(CacheMissReason::NotFound)
    ));
    assert!(matches!(publication, CachePublication::Published(_)));
    assert!(matches!(after, CacheLookup::Hit(_)));
    Ok(())
}

struct BlockingReader {
    bytes: Option<&'static [u8]>,
    started: Sender<()>,
    release: Receiver<()>,
}

impl Read for BlockingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let Some(bytes) = self.bytes.take() else {
            return Ok(0);
        };
        self.started
            .send(())
            .map_err(|_disconnected| io::Error::other("reader start receiver disconnected"))?;
        self.release
            .recv()
            .map_err(|_disconnected| io::Error::other("reader release sender disconnected"))?;
        buffer[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }
}
