use crate::level3::sgemm_nn::sgemm_nn; 
use crate::level3::sgemm::{MC, NC}; 
use crate::fused::sscalf; 
use crate::level3::substitutions::{
    backward_sub_panel_left_lower_t, 
    forward_sub_panel_right_lower_t, 
}; 
use crate::types::MatrixMut; 

pub(crate) fn strlsm_left_trans (
    m: usize,
    n: usize,
    alpha: f32,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }

    let bview = MatrixMut::new(b, m, n, ldb, 0)
        .expect("MatrixMut::new failed");
    sscalf(alpha, bview);

    let mc = MC;
    let mb = m.div_ceil(mc);

    let mut bi = mb;
    while bi > 0 {
        bi -= 1;

        let i0 = bi * mc;
        let ib_dim = mc.min(m - i0);

        let a_ii = &a[i0 + i0 * lda..];
        let b_i  = &mut b[i0..];

        backward_sub_panel_left_lower_t (
            ib_dim,
            n,
            unit_diag,
            a_ii,
            lda,
            b_i,
            ldb,
        );

        // for now, I'll use a vec to avoid aliasing and 
        // keep things safe until I think of a workaround.
        // for small matrices this should only 
        // reduce performance a few percent (O(1/m)). 
        let mut bi_buf = vec![0.0; ib_dim * n];

        // copy solved panel b_i into temporary buf 
        for j in 0..n {
            let src_col = &b[i0 + j * ldb .. i0 + j * ldb + ib_dim];
            let dst_col = &mut bi_buf[j * ib_dim .. (j + 1) * ib_dim];
            dst_col.copy_from_slice(src_col);
        }

        let mut bk = 0;
        while bk < bi {
            let k0 = bk * mc;
            let kb_dim = mc.min(m - k0);

            let mut a_t_buf = vec![0.0; kb_dim * ib_dim];

            for q in 0..ib_dim {
                let row = i0 + q;
                for p in 0..kb_dim {
                    let col = k0 + p;
                    a_t_buf[p + q * kb_dim] = a[row + col * lda];
                }
            }

            let a_t = a_t_buf.as_slice();
            let x_i = bi_buf.as_slice();
            let b_k = &mut b[k0..];

            let lda_at = kb_dim;
            let ldb_xi = ib_dim;

            sgemm_nn(
                kb_dim,
                n,
                ib_dim,
                -1.0,
                a_t,
                lda_at,
                x_i,
                ldb_xi,
                1.0,
                b_k,
                ldb,
            );

            bk += 1;
        }
    }
}

pub(crate) fn strlsm_right_trans (
    m: usize,
    n: usize,
    alpha: f32,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }

    let bview = MatrixMut::new(b, m, n, ldb, 0)
        .expect("MatrixMut::new failed");
    sscalf(alpha, bview);

    let nc = NC;
    let nb = n.div_ceil(nc);

    for bj in 0..nb {
        let j0 = bj * nc;
        let jb_dim = nc.min(n - j0);

        let a_ii = &a[j0 + j0 * lda..];

        let split = j0 * ldb;
        let (_b_left, b_j_and_right) = b.split_at_mut(split);
        let (b_j, b_right) = b_j_and_right.split_at_mut(jb_dim * ldb);

        forward_sub_panel_right_lower_t (
            m,
            jb_dim,
            unit_diag,
            a_ii,
            lda,
            b_j,
            ldb,
        );

        let x_j: &[f32] = b_j;

        let mut bk = bj + 1;
        while bk < nb {
            let k0 = bk * nc;
            let kb_dim = nc.min(n - k0);

            let mut a_t_buf = vec![0.0; jb_dim * kb_dim];
            let a_kj = &a[k0 + j0 * lda..];

            let mut p = 0;
            while p < jb_dim {
                let mut q = 0;
                while q < kb_dim {
                    a_t_buf[p + q * jb_dim] = a_kj[q + p * lda];
                    q += 1;
                }
                p += 1;
            }

            let col0_rel = k0 - (j0 + jb_dim);
            let b_k = &mut b_right[col0_rel * ldb .. (col0_rel + kb_dim) * ldb];

            sgemm_nn(
                m,
                kb_dim,
                jb_dim,
                -1.0,
                x_j,
                ldb,
                a_t_buf.as_slice(),
                jb_dim,
                1.0,
                b_k,
                ldb,
            );

            bk += 1;
        }
    }
}
