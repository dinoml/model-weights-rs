//! Runtime-gated bulk BF16 conversion. Scalar callers retain the same byte contract.
use crate::{CancellationToken, Result, tensor::DType};
#[cfg(target_arch = "x86_64")]
use crate::{Error, ErrorCategory};

#[cfg_attr(
    target_arch = "x86_64",
    expect(
        unsafe_code,
        reason = "runtime feature detection gates the AVX2/F16C conversion"
    )
)]
pub(crate) fn cast_to_bf16(
    dtype: DType,
    source: &[u8],
    output_len: usize,
    cancellation: &CancellationToken,
) -> Result<Option<Box<[u8]>>> {
    #[cfg(target_arch = "x86_64")]
    if matches!(dtype, DType::F16 | DType::F32)
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        let width = if dtype == DType::F16 { 2 } else { 4 };
        if source.len() % width != 0
            || output_len / 2 != source.len() / width
            || output_len % 2 != 0
        {
            return Err(Error::integrity("BF16 conversion buffer length mismatch"));
        }
        cancellation.check()?;
        let mut output = Vec::new();
        output.try_reserve_exact(output_len).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "BF16 conversion allocation failed",
                source,
            )
        })?;
        output.resize(output_len, 0);
        for (input, destination) in source
            .chunks(16 * 1024 * width)
            .zip(output.chunks_mut(32 * 1024))
        {
            cancellation.check()?;
            // SAFETY: both features were detected above. The bounded slices have
            // matching element counts; the kernel bounds every vector load/store.
            unsafe { convert_block(input, destination, dtype == DType::F16) };
        }
        cancellation.check()?;
        return Ok(Some(output.into_boxed_slice()));
    }
    let _ = (dtype, source, output_len, cancellation);
    Ok(None)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[expect(
    unsafe_code,
    reason = "unaligned SIMD loads/stores are bounded by exact 8-element chunks"
)]
unsafe fn convert_block(source: &[u8], destination: &mut [u8], half_source: bool) {
    use std::arch::x86_64::{
        _mm_loadu_si128, _mm_packus_epi32, _mm_storeu_si128, _mm256_add_epi32, _mm256_and_si256,
        _mm256_blendv_epi8, _mm256_castps_si256, _mm256_castsi256_si128, _mm256_cmpgt_epi32,
        _mm256_cvtph_ps, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_or_si256,
        _mm256_set1_epi32, _mm256_srli_epi32,
    };
    let width = if half_source { 2 } else { 4 };
    let count = destination.len() / 2;
    let vectors = count / 8;
    for (input, output) in source[..vectors * 8 * width]
        .chunks_exact(8 * width)
        .zip(destination[..vectors * 16].chunks_exact_mut(16))
    {
        // SAFETY: input is exactly 16 (FP16) or 32 (FP32) bytes and output is
        // exactly 16 bytes. Unaligned intrinsics impose no alignment requirement.
        unsafe {
            let bits = if half_source {
                _mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(input.as_ptr().cast())))
            } else {
                _mm256_loadu_si256(input.as_ptr().cast())
            };
            let high = _mm256_srli_epi32::<16>(bits);
            let lsb = _mm256_and_si256(high, _mm256_set1_epi32(1));
            let rounded = _mm256_srli_epi32::<16>(_mm256_add_epi32(
                bits,
                _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb),
            ));
            let nan = _mm256_cmpgt_epi32(
                _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff_ffff)),
                _mm256_set1_epi32(0x7f80_0000),
            );
            let result =
                _mm256_blendv_epi8(rounded, _mm256_or_si256(high, _mm256_set1_epi32(0x40)), nan);
            let packed = _mm_packus_epi32(
                _mm256_castsi256_si128(result),
                _mm256_extracti128_si256::<1>(result),
            );
            _mm_storeu_si128(output.as_mut_ptr().cast(), packed);
        }
    }
    for (input, output) in source[vectors * 8 * width..]
        .chunks_exact(width)
        .zip(destination[vectors * 16..].chunks_exact_mut(2))
    {
        let value = if half_source {
            half::f16::from_bits(u16::from_le_bytes([input[0], input[1]])).to_f32()
        } else {
            f32::from_le_bytes([input[0], input[1], input[2], input[3]])
        };
        output.copy_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
    }
}
