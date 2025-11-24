use std::cmp::min;
use crate::level3::{
    sgemm::{MC, NC, KC},
    macrokernel::macrokernel,
    packers::{
        pack_a_block_t, pack_b_block_t,
        a_buf_len, b_buf_len,
    },
};

pub(crate) fn sgemm_tt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    debug_assert!(
        ldc >= m && lda >= k && ldb >= n,
        "matrix dimensions don't satisfy lda/ldb/ldc"
    );

    if n > 0 {
        debug_assert!(c.len() >= (n - 1) * ldc + m);
    }
    if m > 0 && k > 0 {
        debug_assert!(a.len() >= (m - 1) * lda + k);
    }
    if n > 0 && k > 0 {
        debug_assert!(b.len() >= (k - 1) * ldb + n);
    }


    // quick return 
    if alpha == 0.0 || k == 0 {
        if beta == 0.0 {
            for j in 0..n {
                let col = &mut c[j * ldc .. j * ldc + m];
                col.fill(0.0);
            }
        } else if beta != 1.0 {
            for j in 0..n {
                let col = &mut c[j * ldc .. j * ldc + m];
                for x in col.iter_mut() {
                    *x *= beta;
                }
            }
        }
        return;
    }

    let mut a_buf = vec![0.0; a_buf_len(MC, KC)];
    let mut b_buf = vec![0.0; b_buf_len(KC, NC)];

    let mut j0 = 0;
    while j0 < n {
        let nc = min(NC, n - j0);

        let mut l0 = 0;
        while l0 < k {
            let kcblk = min(KC, k - l0);

            // pack B^T `kcblk x nc` starting at (l0, j0) 
            {
                let b_block_offset = j0 + l0 * ldb;
                pack_b_block_t(
                    kcblk,
                    nc,
                    b,
                    ldb,
                    b_block_offset,
                    &mut b_buf,
                );
            }

            let beta_panel = if l0 == 0 { beta } else { 1.0 };

            let mut i0 = 0;
            while i0 < m {
                let mc = min(MC, m - i0);

                // pack A^T `mc x kcblk` starting at (l0, i0) 
                {
                    let a_block_offset = l0 + i0 * lda;
                    pack_a_block_t(
                        mc,
                        kcblk,
                        a,
                        lda,
                        a_block_offset,
                        &mut a_buf,
                    );
                }

                // base of C block at (i0, j0)
                let c_offset = i0 + j0 * ldc;
                let c_block = &mut c[c_offset..];

                macrokernel(
                    mc,
                    nc,
                    kcblk,
                    alpha,
                    beta_panel,
                    &a_buf,
                    &b_buf,
                    c_block,
                    ldc,
                );

                i0 += mc;
            }

            l0 += kcblk;
        }

        j0 += nc;
    }
}

