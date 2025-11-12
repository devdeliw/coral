use crate::level3::{
    zgemm::{MC, NC, KC},
    c64_macro_kernel::macro_kernel,
    c64_packers::{
        pack_a_block_ct,
        pack_b_block_t,
        a_buf_len, 
        b_buf_len,
    },
};
use crate::level3::microkernel::c64_mrxnr::Complex64;

#[inline(always)]
fn is_zero(z: Complex64) -> bool {
    z.re == 0.0 && z.im == 0.0
}

const ONE_Z : Complex64 = Complex64 {
    re: 1.0, 
    im: 0.0
};

pub(crate) fn zgemm_ct(
    m     : usize,
    n     : usize,
    k     : usize,
    alpha : Complex64,
    a     : *const f64,
    lda   : usize,
    b     : *const f64,
    ldb   : usize,
    beta  : Complex64,
    c     : *mut f64,
    ldc   : usize,
) {
    debug_assert!(
        ldc >= m && lda >= k && ldb >= n,
        "matrix dimensions don't satisfy lda/ldb/ldc (in complex elements)"
    );

    unsafe {
        if is_zero(alpha) || k == 0 {
            if is_zero(beta) {
                for j in 0..n {
                    let col = c.add(2 * j * ldc);
                    core::ptr::write_bytes(col, 0, 2 * m);
                }
            } else if beta.re == 1.0 && beta.im == 0.0 {
            } else {
                let br = beta.re;
                let bi = beta.im;

                for j in 0..n {
                    let col = c.add(2 * j * ldc);

                    for i in 0..m {
                        let re = *col.add(2 * i + 0);
                        let im = *col.add(2 * i + 1);

                        *col.add(2 * i + 0) = re * br - im * bi;
                        *col.add(2 * i + 1) = re * bi + im * br;
                    }
                }
            }
            return;
        }

        let mut a_buf = vec![0.0; a_buf_len(MC, KC)];
        let mut b_buf = vec![0.0; b_buf_len(KC, NC)];

        let mut j0 = 0;
        while j0 < n {
            let nc = core::cmp::min(NC, n - j0);

            let mut l0 = 0;
            while l0 < k {
                let kcblk = core::cmp::min(KC, k - l0);

                {
                    let b_block_base = b.add(2 * (j0 + l0 * ldb));
                    pack_b_block_t(
                        kcblk,
                        nc,
                        b_block_base,
                        ldb,
                        b_buf.as_mut_ptr(),
                    );
                }

                let beta_panel = if l0 == 0 { beta } else { ONE_Z };

                let mut i0 = 0;
                while i0 < m {
                    let mc = core::cmp::min(MC, m - i0);

                    {
                        let a_block_base = a.add(2 * (l0 + i0 * lda));
                        pack_a_block_ct(
                            mc,
                            kcblk,
                            a_block_base,
                            lda,
                            a_buf.as_mut_ptr(),
                        );
                    }

                    let c_base = c.add(2 * (i0 + j0 * ldc));

                    macro_kernel(
                        mc,
                        nc,
                        kcblk,
                        alpha,
                        beta_panel,
                        a_buf.as_ptr(),
                        b_buf.as_ptr(),
                        c_base,
                        ldc,
                    );

                    i0 += mc;
                }

                l0 += kcblk;
            }

            j0 += nc;
        }
    }
}

