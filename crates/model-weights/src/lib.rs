//! Inventories, plans, prepares, caches, and delivers immutable model weights.
//!
//! `model-weights` sits between repository/configuration discovery and a model
//! runtime. It deliberately does not perform network acquisition, parse model
//! configuration formats, compile graphs, place tensors on devices, or execute
//! inference.
//!
//! The crate accepts ordinary local files and retained immutable snapshots. A
//! trusted snapshot digest may be reused without reading the full checkpoint;
//! ordinary local files are hashed on demand before content-addressed plans or
//! cache entries are created.

#![cfg_attr(docsrs, feature(doc_cfg))]

mod cancel;
mod checkpoint;
mod error;

pub mod cache;
pub mod identity;
pub mod inventory;
pub mod limits;
pub mod materialize;
pub mod operation;
mod operation_simd;
pub mod overlay;
pub mod pipeline;
pub mod plan;
pub mod prepare;
pub mod quantization;
pub mod source;
pub mod telemetry;
pub mod tensor;

#[doc(inline)]
pub use cancel::CancellationToken;
#[doc(inline)]
pub use checkpoint::{
    AccessMode, Checkpoint, CheckpointBuilder, PlainTensor, QuantizedTensor, TensorData,
};
#[doc(inline)]
pub use error::{Error, ErrorCategory, Result};
