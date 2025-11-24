use crate::level3::packers::{MR, NR};
use std::simd::{Simd, StdFloat};

type Vf32 = Simd<f32, MR>;

#[inline(always)]
pub(crate) fn mrxnr_beta0(
    kc: usize,
    a: &[f32],   
    b: &[f32],   
    c: &mut [f32],
    ldc: usize,
    alpha: f32,
) {
    debug_assert!(a.len() >= kc * MR);
    debug_assert!(b.len() >= kc * NR);
    debug_assert!(c.len() >= (NR - 1) * ldc + MR);

    let (a_chunks, a_tail) = a.as_chunks::<MR>();
    let (b_chunks, b_tail) = b.as_chunks::<NR>();
    debug_assert!(a_tail.is_empty());
    debug_assert!(b_tail.is_empty());
    debug_assert!(a_chunks.len() >= kc && b_chunks.len() >= kc);

    let mut acc: [Vf32; NR] = [Simd::splat(0.0); NR];

    for k in 0..kc {
        let a_vec = Vf32::from_array(a_chunks[k]);   
        let b_row = b_chunks[k];                     

        for j in 0..NR {
            let bj = Simd::splat(b_row[j]);
            acc[j] = a_vec.mul_add(bj, acc[j]);
        }
    }

    let alpha_v = Simd::splat(alpha);

    // beta = 0: C := alpha * acc
    for (acc_col, c_col_full) in acc.iter().zip(c.chunks_exact_mut(ldc)) {
        let c_col = &mut c_col_full[..MR];

        let v = *acc_col * alpha_v;
        v.copy_to_slice(c_col);
    }
}

#[inline(always)]
pub(crate) fn mrxnr_beta1(
    kc: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
    alpha: f32,
) {
    debug_assert!(a.len() >= kc * MR);
    debug_assert!(b.len() >= kc * NR);
    debug_assert!(c.len() >= (NR - 1) * ldc + MR);

    let (a_chunks, a_tail) = a.as_chunks::<MR>();
    let (b_chunks, b_tail) = b.as_chunks::<NR>();
    debug_assert!(a_tail.is_empty());
    debug_assert!(b_tail.is_empty());
    debug_assert!(a_chunks.len() >= kc && b_chunks.len() >= kc);

    let mut acc: [Vf32; NR] = [Simd::splat(0.0); NR];

    for k in 0..kc {
        let a_vec = Vf32::from_array(a_chunks[k]);
        let b_row = b_chunks[k];

        for j in 0..NR {
            let bj = Simd::splat(b_row[j]);
            acc[j] = a_vec.mul_add(bj, acc[j]);
        }
    }

    let alpha_v = Simd::splat(alpha);

    // beta = 1: C += alpha * acc
    for (acc_col, c_col_full) in acc.iter().zip(c.chunks_exact_mut(ldc)) {
        let c_col = &mut c_col_full[..MR];

        let mut c_vec = Vf32::from_slice(c_col);
        c_vec = acc_col.mul_add(alpha_v, c_vec);
        c_vec.copy_to_slice(c_col);
    }
}

#[inline(always)]
pub(crate) fn mrxnr_betax(
    kc: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert!(a.len() >= kc * MR);
    debug_assert!(b.len() >= kc * NR);
    debug_assert!(c.len() >= (NR - 1) * ldc + MR);

    let (a_chunks, a_tail) = a.as_chunks::<MR>();
    let (b_chunks, b_tail) = b.as_chunks::<NR>();
    debug_assert!(a_tail.is_empty());
    debug_assert!(b_tail.is_empty());
    debug_assert!(a_chunks.len() >= kc && b_chunks.len() >= kc);

    let mut acc: [Vf32; NR] = [Simd::splat(0.0); NR];

    for k in 0..kc {
        let a_vec = Vf32::from_array(a_chunks[k]);
        let b_row = b_chunks[k];

        for j in 0..NR {
            let bj = Simd::splat(b_row[j]);
            acc[j] = a_vec.mul_add(bj, acc[j]);
        }
    }

    let alpha_v = Simd::splat(alpha);
    let beta_v  = Simd::splat(beta);

    // beta general: C := beta*C + alpha*acc
    for (acc_col, c_col_full) in acc.iter().zip(c.chunks_exact_mut(ldc)) {
        let c_col = &mut c_col_full[..MR];

        let mut c_vec = Vf32::from_slice(c_col);
        let scaled_c = c_vec * beta_v;
        c_vec = acc_col.mul_add(alpha_v, scaled_c);
        c_vec.copy_to_slice(c_col);
    }
}

