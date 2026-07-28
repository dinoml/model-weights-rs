//! Architecture-specific host kernels for structural tensor operations.

use crate::{CancellationToken, Error, Result};

const SPATIAL_ELEMENTS: usize = 9;
const ELEMENT_BYTES: usize = 2;
const SOURCE_CHANNEL_BYTES: usize = SPATIAL_ELEMENTS * ELEMENT_BYTES;
const SIMD_CHANNELS: usize = 16;
#[cfg(target_arch = "x86_64")]
const SIMD_SOURCE_BYTES: usize = SIMD_CHANNELS * SOURCE_CHANNEL_BYTES;
#[cfg(target_arch = "x86_64")]
const SIMD_OUTPUT_BYTES: usize = SIMD_CHANNELS * ELEMENT_BYTES;
// Keep cancellation latency below the generic structural-operation bound of
// 16K copied elements while ending every non-final chunk on a SIMD boundary.
const CANCELLATION_CHANNELS: usize =
    SIMD_CHANNELS * ((16 * 1024) / (SPATIAL_ELEMENTS * SIMD_CHANNELS));

/// Returns whether the specialized host kernel can run on this processor.
#[must_use]
pub(crate) fn oihw_to_ohwi_3x3_u16_avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Copies one dense 3x3 OIHW tensor into OHWI storage as opaque u16 elements.
pub(crate) fn permute_oihw_to_ohwi_3x3_u16(
    input: &[u8],
    output: &mut [u8],
    output_channels: usize,
    input_channels: usize,
    cancellation: &CancellationToken,
) -> Result<()> {
    permute_oihw_to_ohwi_3x3_u16_with_dispatch(
        input,
        output,
        output_channels,
        input_channels,
        cancellation,
    )
    .map(|_| ())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dispatch {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

#[cfg_attr(
    target_arch = "x86_64",
    expect(
        unsafe_code,
        reason = "calling a runtime-gated AVX2 target-feature function"
    )
)]
fn permute_oihw_to_ohwi_3x3_u16_with_dispatch(
    input: &[u8],
    output: &mut [u8],
    output_channels: usize,
    input_channels: usize,
    cancellation: &CancellationToken,
) -> Result<Dispatch> {
    let (output_channel_bytes, expected_bytes) = checked_lengths(output_channels, input_channels)?;
    if input.len() != expected_bytes || output.len() != expected_bytes {
        return Err(Error::integrity(
            "3x3 u16 OIHW-to-OHWI buffers differ from their inferred byte length",
        ));
    }
    cancellation.check()?;
    if expected_bytes == 0 {
        return Ok(Dispatch::Scalar);
    }

    #[cfg(target_arch = "x86_64")]
    if oihw_to_ohwi_3x3_u16_avx2_available() {
        // SAFETY: Runtime detection proves AVX2 support. The function retains
        // safe slice bounds for all addressing and checks cancellation between
        // bounded chunks.
        unsafe {
            permute_avx2(
                input,
                output,
                output_channels,
                input_channels,
                output_channel_bytes,
                cancellation,
            )?;
            return Ok(Dispatch::Avx2);
        }
    }

    permute_scalar(
        input,
        output,
        output_channels,
        input_channels,
        output_channel_bytes,
        cancellation,
    )?;
    Ok(Dispatch::Scalar)
}

fn checked_lengths(output_channels: usize, input_channels: usize) -> Result<(usize, usize)> {
    let output_channel_bytes = input_channels
        .checked_mul(SOURCE_CHANNEL_BYTES)
        .ok_or_else(|| Error::limit("3x3 u16 OIHW channel byte length overflows usize"))?;
    let expected_bytes = output_channels
        .checked_mul(output_channel_bytes)
        .ok_or_else(|| Error::limit("3x3 u16 OIHW byte length overflows usize"))?;
    Ok((output_channel_bytes, expected_bytes))
}

fn permute_scalar(
    input: &[u8],
    output: &mut [u8],
    output_channels: usize,
    input_channels: usize,
    output_channel_bytes: usize,
    cancellation: &CancellationToken,
) -> Result<()> {
    for output_channel in 0..output_channels {
        let (source_channel, output_channel) =
            channel_slices(input, output, output_channel, output_channel_bytes)?;
        let mut output_rows = OutputRows::new(output_channel, input_channels)?;
        let mut channel_start = 0;
        while channel_start < input_channels {
            cancellation.check()?;
            let channel_end = channel_start
                .saturating_add(CANCELLATION_CHANNELS)
                .min(input_channels);
            output_rows.copy_scalar_channels(source_channel, channel_start, channel_end)?;
            channel_start = channel_end;
        }
    }
    cancellation.check()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(
    unsafe_code,
    reason = "the caller runtime-gates this AVX2 target-feature function"
)]
/// Runs the bounded permutation loop with AVX2 block dispatch.
///
/// # Safety
///
/// The current processor must support AVX2. The caller establishes this with
/// runtime feature detection before entering the function.
unsafe fn permute_avx2(
    input: &[u8],
    output: &mut [u8],
    output_channels: usize,
    input_channels: usize,
    output_channel_bytes: usize,
    cancellation: &CancellationToken,
) -> Result<()> {
    for output_channel in 0..output_channels {
        let (source_channel, output_channel) =
            channel_slices(input, output, output_channel, output_channel_bytes)?;
        let mut output_rows = OutputRows::new(output_channel, input_channels)?;
        let mut channel_start = 0;
        while channel_start < input_channels {
            cancellation.check()?;
            let channel_end = channel_start
                .saturating_add(CANCELLATION_CHANNELS)
                .min(input_channels);
            let mut input_channel = channel_start;
            while input_channel + SIMD_CHANNELS <= channel_end {
                let source_start = input_channel
                    .checked_mul(SOURCE_CHANNEL_BYTES)
                    .ok_or_else(|| Error::limit("SIMD OIHW source offset overflows usize"))?;
                let source_end = source_start
                    .checked_add(SIMD_SOURCE_BYTES)
                    .ok_or_else(|| Error::limit("SIMD OIHW source end overflows usize"))?;
                let source_block = source_channel
                    .get(source_start..source_end)
                    .and_then(<[u8]>::first_chunk::<SIMD_SOURCE_BYTES>)
                    .ok_or_else(|| Error::integrity("SIMD OIHW source block is out of bounds"))?;
                let output_block = output_rows.block_16(input_channel)?;
                // SAFETY: This function is AVX2-enabled. `source_block` is an
                // exact sixteen-channel-by-nine-element block, and
                // `OutputRows::block_16` returns eight disjoint exact 32-byte
                // destinations selected entirely through safe slice APIs.
                unsafe { transpose_16_channels_8_spatial(source_block, output_block) };
                output_rows.copy_scalar_ninth(source_channel, input_channel)?;
                input_channel += SIMD_CHANNELS;
            }
            output_rows.copy_scalar_channels(source_channel, input_channel, channel_end)?;
            channel_start = channel_end;
        }
    }
    cancellation.check()
}

fn channel_slices<'a>(
    input: &'a [u8],
    output: &'a mut [u8],
    output_channel: usize,
    output_channel_bytes: usize,
) -> Result<(&'a [u8], &'a mut [u8])> {
    let start = output_channel
        .checked_mul(output_channel_bytes)
        .ok_or_else(|| Error::limit("OIHW output-channel offset overflows usize"))?;
    let end = start
        .checked_add(output_channel_bytes)
        .ok_or_else(|| Error::limit("OIHW output-channel end overflows usize"))?;
    let source_channel = input
        .get(start..end)
        .ok_or_else(|| Error::integrity("OIHW source channel is out of bounds"))?;
    let output_channel = output
        .get_mut(start..end)
        .ok_or_else(|| Error::integrity("OHWI output channel is out of bounds"))?;
    Ok((source_channel, output_channel))
}

struct OutputRows<'a> {
    rows: [&'a mut [u8]; SPATIAL_ELEMENTS],
}

impl<'a> OutputRows<'a> {
    fn new(output_channel: &'a mut [u8], input_channels: usize) -> Result<Self> {
        let row_bytes = input_channels
            .checked_mul(ELEMENT_BYTES)
            .ok_or_else(|| Error::limit("OHWI row byte length overflows usize"))?;
        let (row0, rest) = split_row(output_channel, row_bytes)?;
        let (row1, rest) = split_row(rest, row_bytes)?;
        let (row2, rest) = split_row(rest, row_bytes)?;
        let (row3, rest) = split_row(rest, row_bytes)?;
        let (row4, rest) = split_row(rest, row_bytes)?;
        let (row5, rest) = split_row(rest, row_bytes)?;
        let (row6, rest) = split_row(rest, row_bytes)?;
        let (row7, rest) = split_row(rest, row_bytes)?;
        let (row8, rest) = split_row(rest, row_bytes)?;
        if !rest.is_empty() {
            return Err(Error::integrity(
                "OHWI output channel contains bytes beyond nine rows",
            ));
        }
        Ok(Self {
            rows: [row0, row1, row2, row3, row4, row5, row6, row7, row8],
        })
    }

    #[cfg(target_arch = "x86_64")]
    fn block_16(&mut self, input_channel: usize) -> Result<OutputBlock16<'_>> {
        let start = input_channel
            .checked_mul(ELEMENT_BYTES)
            .ok_or_else(|| Error::limit("SIMD OHWI output offset overflows usize"))?;
        let end = start
            .checked_add(SIMD_OUTPUT_BYTES)
            .ok_or_else(|| Error::limit("SIMD OHWI output end overflows usize"))?;
        let [row0, row1, row2, row3, row4, row5, row6, row7, _row8] = &mut self.rows;
        Ok(OutputBlock16 {
            rows: [
                exact_output_block(row0, start, end)?,
                exact_output_block(row1, start, end)?,
                exact_output_block(row2, start, end)?,
                exact_output_block(row3, start, end)?,
                exact_output_block(row4, start, end)?,
                exact_output_block(row5, start, end)?,
                exact_output_block(row6, start, end)?,
                exact_output_block(row7, start, end)?,
            ],
        })
    }

    #[cfg(target_arch = "x86_64")]
    fn copy_scalar_ninth(&mut self, source_channel: &[u8], input_channel: usize) -> Result<()> {
        self.copy_scalar_row(
            source_channel,
            input_channel,
            input_channel + SIMD_CHANNELS,
            SPATIAL_ELEMENTS - 1,
        )
    }

    fn copy_scalar_channels(
        &mut self,
        source_channel: &[u8],
        channel_start: usize,
        channel_end: usize,
    ) -> Result<()> {
        for spatial_index in 0..SPATIAL_ELEMENTS {
            self.copy_scalar_row(source_channel, channel_start, channel_end, spatial_index)?;
        }
        Ok(())
    }

    fn copy_scalar_row(
        &mut self,
        source_channel: &[u8],
        channel_start: usize,
        channel_end: usize,
        spatial_index: usize,
    ) -> Result<()> {
        let output_row = self
            .rows
            .get_mut(spatial_index)
            .ok_or_else(|| Error::integrity("OHWI spatial row is out of bounds"))?;
        for input_channel in channel_start..channel_end {
            let source_start = input_channel
                .checked_mul(SOURCE_CHANNEL_BYTES)
                .and_then(|offset| {
                    spatial_index
                        .checked_mul(ELEMENT_BYTES)
                        .and_then(|spatial| offset.checked_add(spatial))
                })
                .ok_or_else(|| Error::limit("scalar OIHW source offset overflows usize"))?;
            let source_end = source_start
                .checked_add(ELEMENT_BYTES)
                .ok_or_else(|| Error::limit("scalar OIHW source end overflows usize"))?;
            let output_start = input_channel
                .checked_mul(ELEMENT_BYTES)
                .ok_or_else(|| Error::limit("scalar OHWI output offset overflows usize"))?;
            let output_end = output_start
                .checked_add(ELEMENT_BYTES)
                .ok_or_else(|| Error::limit("scalar OHWI output end overflows usize"))?;
            let source = source_channel
                .get(source_start..source_end)
                .ok_or_else(|| Error::integrity("scalar OIHW source element is out of bounds"))?;
            let target = output_row
                .get_mut(output_start..output_end)
                .ok_or_else(|| Error::integrity("scalar OHWI output element is out of bounds"))?;
            target.copy_from_slice(source);
        }
        Ok(())
    }
}

fn split_row(output: &mut [u8], row_bytes: usize) -> Result<(&mut [u8], &mut [u8])> {
    if output.len() < row_bytes {
        return Err(Error::integrity("OHWI output row is out of bounds"));
    }
    Ok(output.split_at_mut(row_bytes))
}

#[cfg(target_arch = "x86_64")]
fn exact_output_block(
    row: &mut [u8],
    start: usize,
    end: usize,
) -> Result<&mut [u8; SIMD_OUTPUT_BYTES]> {
    row.get_mut(start..end)
        .and_then(|bytes| bytes.first_chunk_mut::<SIMD_OUTPUT_BYTES>())
        .ok_or_else(|| Error::integrity("SIMD OHWI output block is out of bounds"))
}

#[cfg(target_arch = "x86_64")]
struct OutputBlock16<'a> {
    rows: [&'a mut [u8; SIMD_OUTPUT_BYTES]; SPATIAL_ELEMENTS - 1],
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(
    unsafe_code,
    reason = "AVX2 unaligned loads and stores are isolated to this exact-block kernel"
)]
#[expect(
    clippy::cast_ptr_alignment,
    reason = "loadu/storeu intrinsics explicitly support unaligned byte slices"
)]
/// Transposes two lane-parallel 8x8 matrices of opaque u16 elements.
///
/// # Safety
///
/// The current processor must support AVX2. Every source and destination range
/// is encoded in an exact-size reference, and the output references must be
/// mutually disjoint.
unsafe fn transpose_16_channels_8_spatial(
    source: &[u8; SIMD_SOURCE_BYTES],
    output: OutputBlock16<'_>,
) {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm256_castsi128_si256, _mm256_inserti128_si256,
        _mm256_storeu_si256, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64,
        _mm256_unpacklo_epi16, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
    };

    let source_row = |channel: usize| {
        let start = channel * SOURCE_CHANNEL_BYTES;
        source
            .get(start..start + SIMD_OUTPUT_BYTES / 2)
            .and_then(<[u8]>::first_chunk::<{ SIMD_OUTPUT_BYTES / 2 }>)
            .expect("a channel prefix is within the exact SIMD source block")
    };
    // SAFETY: The target-feature contract guarantees AVX2. Every source
    // reference contains exactly sixteen readable bytes; every destination
    // reference contains exactly 32 writable bytes and was derived through
    // disjoint safe row splitting. The loadu/storeu intrinsics explicitly
    // permit unaligned addresses. The remaining intrinsics operate only on
    // register values.
    unsafe {
        let rows = std::array::from_fn::<_, SIMD_CHANNELS, _>(|channel| {
            _mm_loadu_si128(source_row(channel).as_ptr().cast::<__m128i>())
        });
        let paired = std::array::from_fn::<_, { SIMD_CHANNELS / 2 }, _>(|row| {
            _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(rows[row]),
                rows[row + SIMD_CHANNELS / 2],
            )
        });

        let t0 = _mm256_unpacklo_epi16(paired[0], paired[1]);
        let t1 = _mm256_unpackhi_epi16(paired[0], paired[1]);
        let t2 = _mm256_unpacklo_epi16(paired[2], paired[3]);
        let t3 = _mm256_unpackhi_epi16(paired[2], paired[3]);
        let t4 = _mm256_unpacklo_epi16(paired[4], paired[5]);
        let t5 = _mm256_unpackhi_epi16(paired[4], paired[5]);
        let t6 = _mm256_unpacklo_epi16(paired[6], paired[7]);
        let t7 = _mm256_unpackhi_epi16(paired[6], paired[7]);
        let u0 = _mm256_unpacklo_epi32(t0, t2);
        let u1 = _mm256_unpackhi_epi32(t0, t2);
        let u2 = _mm256_unpacklo_epi32(t4, t6);
        let u3 = _mm256_unpackhi_epi32(t4, t6);
        let u4 = _mm256_unpacklo_epi32(t1, t3);
        let u5 = _mm256_unpackhi_epi32(t1, t3);
        let u6 = _mm256_unpacklo_epi32(t5, t7);
        let u7 = _mm256_unpackhi_epi32(t5, t7);
        let columns = [
            _mm256_unpacklo_epi64(u0, u2),
            _mm256_unpackhi_epi64(u0, u2),
            _mm256_unpacklo_epi64(u1, u3),
            _mm256_unpackhi_epi64(u1, u3),
            _mm256_unpacklo_epi64(u4, u6),
            _mm256_unpackhi_epi64(u4, u6),
            _mm256_unpacklo_epi64(u5, u7),
            _mm256_unpackhi_epi64(u5, u7),
        ];

        for (target, column) in output.rows.into_iter().zip(columns) {
            _mm256_storeu_si256(target.as_mut_ptr().cast::<__m256i>(), column);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(input: &[u8], output_channels: usize, input_channels: usize) -> Vec<u8> {
        let mut output = Vec::with_capacity(input.len());
        for output_channel in 0..output_channels {
            for spatial_index in 0..SPATIAL_ELEMENTS {
                for input_channel in 0..input_channels {
                    let source_element = (output_channel * input_channels + input_channel)
                        * SPATIAL_ELEMENTS
                        + spatial_index;
                    let source_start = source_element * ELEMENT_BYTES;
                    output.extend_from_slice(&input[source_start..source_start + ELEMENT_BYTES]);
                }
            }
        }
        output
    }

    #[test]
    fn raw_u16_bit_patterns_are_preserved_exhaustively() -> Result<()> {
        let (output_channels, input_channels) = (1_usize, 1_usize << 16);
        let mut input = Vec::with_capacity(output_channels * input_channels * SOURCE_CHANNEL_BYTES);
        for bits in 0_u16..=u16::MAX {
            for spatial_index in 0..SPATIAL_ELEMENTS {
                let value =
                    bits.rotate_left(u32::try_from(spatial_index).expect("spatial index fits u32"));
                input.extend_from_slice(&value.to_ne_bytes());
            }
        }
        let expected = reference(&input, output_channels, input_channels);
        let mut actual = vec![0_u8; input.len()];

        permute_oihw_to_ohwi_3x3_u16(
            &input,
            &mut actual,
            output_channels,
            input_channels,
            &CancellationToken::new(),
        )?;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn channel_boundaries_match_portable_reference() -> Result<()> {
        for input_channels in [
            0_usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1_279, 1_280, 1_281,
        ] {
            let output_channels = 2;
            let input = (0..output_channels * input_channels * SOURCE_CHANNEL_BYTES)
                .map(|index| {
                    u8::try_from(index.wrapping_mul(73).wrapping_add(index.rotate_left(5)) % 251)
                        .expect("value is below 251")
                })
                .collect::<Vec<_>>();
            let expected = reference(&input, output_channels, input_channels);
            let mut actual = vec![0_u8; input.len()];

            permute_oihw_to_ohwi_3x3_u16(
                &input,
                &mut actual,
                output_channels,
                input_channels,
                &CancellationToken::new(),
            )?;

            assert_eq!(actual, expected, "input-channel boundary {input_channels}");
        }
        Ok(())
    }

    #[test]
    fn runtime_dispatch_uses_avx2_when_available() -> Result<()> {
        let (output_channels, input_channels) = (1_usize, SIMD_CHANNELS);
        let input = vec![0xA5_u8; output_channels * input_channels * SOURCE_CHANNEL_BYTES];
        let mut output = vec![0_u8; input.len()];
        let dispatch = permute_oihw_to_ohwi_3x3_u16_with_dispatch(
            &input,
            &mut output,
            output_channels,
            input_channels,
            &CancellationToken::new(),
        )?;

        #[cfg(target_arch = "x86_64")]
        if oihw_to_ohwi_3x3_u16_avx2_available() {
            assert_eq!(dispatch, Dispatch::Avx2);
        } else {
            assert_eq!(dispatch, Dispatch::Scalar);
        }
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(dispatch, Dispatch::Scalar);
        Ok(())
    }

    #[test]
    fn cancelled_work_stops_before_dispatch() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let input = vec![0_u8; SIMD_CHANNELS * SOURCE_CHANNEL_BYTES];
        let mut output = vec![0_u8; input.len()];

        let error =
            permute_oihw_to_ohwi_3x3_u16(&input, &mut output, 1, SIMD_CHANNELS, &cancellation)
                .expect_err("pre-cancelled SIMD permutation must stop");

        assert_eq!(error.category(), crate::ErrorCategory::Cancelled);
    }
}
