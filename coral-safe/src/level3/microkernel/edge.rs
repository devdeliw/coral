use crate::level3::packers::{MR, NR};

/// scalar edge kernel for partial tiles (mr <= MR, nr <= NR).
///
/// - `a` is the packed A panel of length at least `kc * MR`.
/// - `b` is the packed B panel of length at least `kc * NR`.
/// - `c` is a tile with leading dimension `ldc`,
#[inline(always)]
pub(crate) fn edge(
    mr: usize,
    nr: usize,
    kc: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert!(mr <= MR && nr <= NR);
    debug_assert!(a.len() >= kc * MR);
    debug_assert!(b.len() >= kc * NR);

    if nr > 0 {
        debug_assert!(c.len() >= (nr - 1) * ldc + mr);
    }

    let mut acc = [[0.0; NR]; MR];

    for k_step in 0..kc {
        let a_block = &a[k_step * MR .. k_step * MR + mr]; 
        let b_block = &b[k_step * NR .. k_step * NR + nr]; 

        for (r, &ar) in a_block.iter().enumerate() {
            let acc_row = &mut acc[r];
            for (ccol, &br) in b_block.iter().enumerate() {
                acc_row[ccol] += ar * br;
            }
        }
    }

    // write back to C 
    if beta == 0.0 {
        for r in 0..mr {
            let acc_row = &acc[r][..nr];
            let c_row = &mut c[r..];

            for (dst, &val) in c_row
                .iter_mut()
                .step_by(ldc)
                .take(nr)
                .zip(acc_row.iter())
            {
                *dst = alpha * val;
            }
        }
    } else if beta == 1.0 {
        for r in 0..mr {
            let acc_row = &acc[r][..nr];
            let c_row = &mut c[r..];

            for (dst, &val) in c_row
                .iter_mut()
                .step_by(ldc)
                .take(nr)
                .zip(acc_row.iter())
            {
                *dst += alpha * val;
            }
        }
    } else {
        for r in 0..mr {
            let acc_row = &acc[r][..nr];
            let c_row = &mut c[r..];

            for (dst, &val) in c_row
                .iter_mut()
                .step_by(ldc)
                .take(nr)
                .zip(acc_row.iter())
            {
                *dst = beta * *dst + alpha * val;
            }
        }
    }
}
