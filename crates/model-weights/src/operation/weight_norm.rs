//! FP32 weight normalization of logical, contiguous checkpoint tensors.

use super::{
    Axis, COPY_TILE_ELEMENTS_USIZE, NodeExecution, OperationNode, RuntimeValue, TensorFacts,
    allocate_tensor, fresh_node_execution, only_output, validate_axis, validate_view_len,
};
use crate::prepare::Representation;
use crate::tensor::DType;
use crate::{CancellationToken, Error, ErrorCategory, Result};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

/// Two-input weight normalization, with gain first and direction second.
///
/// The selected dimension is retained; all other dimensions are reduced.
/// Inputs may independently use F16, BF16 or F32. The output is always F32.
/// No epsilon is added: zero norms retain IEEE division semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WeightNorm {
    dim: Axis,
}

impl WeightNorm {
    /// Creates normalization retaining the selected logical axis.
    #[must_use]
    pub const fn new(dim: Axis) -> Self {
        Self { dim }
    }

    /// Returns the retained logical axis.
    #[must_use]
    pub const fn dim(self) -> Axis {
        self.dim
    }
}

pub(super) fn infer_weight_norm(
    operation: WeightNorm,
    inputs: &[&TensorFacts],
) -> Result<Box<[TensorFacts]>> {
    let [g, v] = inputs else {
        return Err(Error::binding(
            "weight normalization requires gain and direction inputs",
        ));
    };
    let axis = validate_axis(operation.dim, v.logical_shape().len())?;
    for input in inputs {
        if !input.is_canonical_contiguous()
            || !matches!(
                input.representation().dtype(),
                DType::F16 | DType::Bf16 | DType::F32
            )
        {
            return Err(Error::binding(
                "weight normalization requires contiguous F16, BF16 or F32 inputs",
            ));
        }
    }
    let mut gain_shape = vec![1; v.logical_shape().len()];
    gain_shape[axis] = v.logical_shape()[axis];
    if g.logical_shape() != gain_shape {
        return Err(Error::binding(
            "weight normalization gain shape must retain only the selected direction axis",
        ));
    }
    Ok(vec![TensorFacts::contiguous(
        v.logical_shape().to_vec(),
        Representation::contiguous(DType::F32),
    )?]
    .into_boxed_slice())
}

pub(super) fn execute_weight_norm(
    operation: WeightNorm,
    node: &OperationNode,
    inputs: &[RuntimeValue],
    facts: &[&TensorFacts],
    cancellation: &CancellationToken,
) -> Result<NodeExecution> {
    let [g, v] = inputs else {
        return Err(Error::integrity(
            "weight normalization requires two runtime inputs",
        ));
    };
    for (input, fact) in inputs.iter().zip(facts) {
        validate_view_len(&input.bytes, fact.byte_len(), "weight normalization input")?;
    }
    let axis = validate_axis(operation.dim, facts[1].logical_shape().len())?;
    let shape = facts[1].logical_shape();
    let channels = usize::try_from(shape[axis]).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "weight normalization axis exceeds usize",
            source,
        )
    })?;
    // TensorFacts has already checked the full product and allocation length.
    let inner = shape[axis + 1..].iter().try_fold(1usize, |n, &d| {
        n.checked_mul(usize::try_from(d).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "weight normalization dimension exceeds usize",
                source,
            )
        })?)
        .ok_or_else(|| Error::limit("weight normalization size overflow"))
    })?;
    let mut output = allocate_tensor(only_output(node)?, cancellation)?;
    let elements = output.len() / 4;
    let lane = channels
        .checked_mul(inner)
        .ok_or_else(|| Error::limit("weight normalization size overflow"))?;
    if lane == 0 {
        return Ok(fresh_node_execution(output));
    }
    let outer = elements / lane;
    for channel in 0..channels {
        cancellation.check()?;
        let mut sum = 0.0f32;
        for block in 0..outer {
            for offset in 0..inner {
                if offset % COPY_TILE_ELEMENTS_USIZE == 0 {
                    cancellation.check()?;
                }
                let index = (block * channels + channel) * inner + offset;
                let value =
                    read_float(v.bytes.as_slice(), facts[1].representation().dtype(), index);
                sum += value * value;
            }
        }
        let scale = read_float(
            g.bytes.as_slice(),
            facts[0].representation().dtype(),
            channel,
        ) / sum.sqrt();
        for block in 0..outer {
            for offset in 0..inner {
                if offset % COPY_TILE_ELEMENTS_USIZE == 0 {
                    cancellation.check()?;
                }
                let index = (block * channels + channel) * inner + offset;
                let value =
                    read_float(v.bytes.as_slice(), facts[1].representation().dtype(), index)
                        * scale;
                output[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(fresh_node_execution(output))
}

fn read_float(bytes: &[u8], dtype: DType, index: usize) -> f32 {
    match dtype {
        DType::F32 => {
            let start = index * 4;
            f32::from_le_bytes([
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
            ])
        }
        DType::F16 => {
            f16::from_bits(u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])).to_f32()
        }
        DType::Bf16 => {
            bf16::from_bits(u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])).to_f32()
        }
        _ => unreachable!("weight normalization dtype validated during inference"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operation::{Operation, OperationGraph, OperationGraphBuilder};
    use crate::plan::TensorName;
    use crate::prepare::PreparationEngine;
    use crate::tensor::ByteView;

    fn graph(dim: u32, g: TensorFacts, v: TensorFacts) -> Result<OperationGraph> {
        let mut graph = OperationGraphBuilder::new();
        let g = graph.add_input(TensorName::parse("g")?, g)?;
        let v = graph.add_input(TensorName::parse("v")?, v)?;
        let operation = Operation::WeightNorm(WeightNorm::new(Axis::from_index(dim)));
        let output =
            graph.add_operation(operation.builtin_implementation()?, operation, [g, v])?[0];
        graph.build(output)
    }

    fn facts(shape: &[u64], dtype: DType) -> Result<TensorFacts> {
        TensorFacts::contiguous(shape.to_vec(), Representation::contiguous(dtype))
    }

    fn bytes(values: &[f32], dtype: DType) -> ByteView {
        let bytes = values
            .iter()
            .flat_map(|&v| match dtype {
                DType::F16 => f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                DType::Bf16 => bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                _ => v.to_le_bytes().to_vec(),
            })
            .collect::<Vec<_>>();
        ByteView::from_boxed(bytes.into_boxed_slice())
    }

    #[test]
    fn mixed_source_dtypes_produce_fp32_and_preserve_ieee_zero_norm() -> Result<()> {
        for g_dtype in [DType::F16, DType::Bf16, DType::F32] {
            for v_dtype in [DType::F16, DType::Bf16, DType::F32] {
                let graph = graph(0, facts(&[3, 1, 1], g_dtype)?, facts(&[3, 1, 2], v_dtype)?)?;
                assert_eq!(graph.output_facts().representation().dtype(), DType::F32);
                let execution = graph.execute_host(
                    &[
                        bytes(&[10., -5., 1.], g_dtype),
                        bytes(&[3., 4., 0., 5., 0., 0.], v_dtype),
                    ],
                    &PreparationEngine::with_builtins()?,
                    &CancellationToken::new(),
                )?;
                let values = execution
                    .output()
                    .as_slice()
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(&values[..4], &[6., 8., -0., -5.]);
                assert!(values[4].is_nan() && values[5].is_nan());
                assert_eq!(
                    execution.peak_scratch_bytes(),
                    graph.estimate_host_scratch_bytes()?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn nonleading_axis_normalizes_across_outer_and_inner_dimensions() -> Result<()> {
        let graph = graph(
            1,
            facts(&[1, 2, 1], DType::F32)?,
            facts(&[2, 2, 2], DType::F32)?,
        )?;
        let wire = serde_json::to_vec(&graph).unwrap();
        let graph: OperationGraph = serde_json::from_slice(&wire).unwrap();
        let execution = graph.execute_host(
            &[
                bytes(&[10., 2.], DType::F32),
                bytes(&[3., 4., 1., 1., 0., 0., 1., 1.], DType::F32),
            ],
            &PreparationEngine::with_builtins()?,
            &CancellationToken::new(),
        )?;
        let values = execution
            .output()
            .as_slice()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, [6., 8., 1., 1., 0., 0., 1., 1.]);
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert_eq!(
            graph
                .execute_host(
                    &[bytes(&[1., 1.], DType::F32), bytes(&[1.; 8], DType::F32)],
                    &PreparationEngine::with_builtins()?,
                    &cancelled
                )
                .unwrap_err()
                .category(),
            ErrorCategory::Cancelled
        );
        Ok(())
    }

    #[test]
    fn fp32_intermediate_is_included_in_storage_preparation_scratch() -> Result<()> {
        use crate::operation::Permute;
        let mut graph = OperationGraphBuilder::new();
        let g = graph.add_input(TensorName::parse("g")?, facts(&[2, 1, 1], DType::F16)?)?;
        let v = graph.add_input(TensorName::parse("v")?, facts(&[2, 1, 2], DType::F16)?)?;
        let norm = Operation::WeightNorm(WeightNorm::new(Axis::from_index(0)));
        let weight = graph.add_operation(norm.builtin_implementation()?, norm, [g, v])?[0];
        let permute = Operation::Permute(Permute::logical([0, 2, 1])?);
        let output = graph.add_operation(permute.builtin_implementation()?, permute, [weight])?[0];
        let graph = graph.build(output)?;
        assert_eq!(graph.estimate_host_scratch_bytes()?, 16);
        let execution = graph.execute_host(
            &[
                bytes(&[10., 5.], DType::F16),
                bytes(&[3., 4., 0., 5.], DType::F16),
            ],
            &PreparationEngine::with_builtins()?,
            &CancellationToken::new(),
        )?;
        assert_eq!(execution.peak_scratch_bytes(), 16);
        Ok(())
    }

    #[test]
    fn invalid_gain_shape_axis_and_dtype_fail_during_inference() -> Result<()> {
        assert_eq!(
            graph(0, facts(&[2], DType::F32)?, facts(&[2, 3, 4], DType::F32)?)
                .unwrap_err()
                .category(),
            ErrorCategory::Binding
        );
        assert_eq!(
            graph(
                3,
                facts(&[2, 1, 1], DType::F32)?,
                facts(&[2, 3, 4], DType::F32)?
            )
            .unwrap_err()
            .category(),
            ErrorCategory::Binding
        );
        assert_eq!(
            graph(
                0,
                facts(&[2, 1, 1], DType::U8)?,
                facts(&[2, 3, 4], DType::F32)?
            )
            .unwrap_err()
            .category(),
            ErrorCategory::Binding
        );
        Ok(())
    }
}
