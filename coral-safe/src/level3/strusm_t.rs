use crate::level3::{sgemm_tn, sgemm_nt};
use crate::level3::sgemm::{MC, NC};
use crate::fused::sscalf;
use crate::level3::subs_upper::{
    forward_sub_panel_left_upper_t, 
    backward_sub_panel_right_upper_t, 
};
use crate::types::MatrixMut;

pub(crate) fn strusm_left_trans (
    m: usize,
    n: usize,
    alpha: f32,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
    scratch: &mut [f32],
) {
    if m == 0 || n == 0 {
        return;
    }

    let bview = MatrixMut::new(b, m, n, ldb, 0)
        .expect("MatrixMut::new failed");
    sscalf(alpha, bview);

    let mc = MC;
    let mb = m.div_ceil(mc);

    for bi in 0..mb {
        let i0 = bi * mc;
        let ib_dim = mc.min(m - i0);

        // solve diagonal 
        let a_ii = &a[i0 + i0 * lda ..];
        let b_i  = &mut b[i0 ..];

        forward_sub_panel_left_upper_t (
            ib_dim,
            n,
            unit_diag,
            a_ii,
            lda,
            b_i,
            ldb,
        );

        let panel_len = ib_dim * n;
        debug_assert!(scratch.len() >= panel_len);
        let (xi_buf, _rest) = scratch.split_at_mut(panel_len);

        // copy solved into scratch buffer
        for j in 0..n {
            let src_col = &b[i0 + j * ldb .. i0 + j * ldb + ib_dim];
            let dst_col = &mut xi_buf[j * ib_dim .. (j + 1) * ib_dim];
            dst_col.copy_from_slice(src_col);
        }

        let x_i: &[f32] = &xi_buf[..];
        let ldx = ib_dim;

        let mut l0 = i0 + ib_dim;
        while l0 < m {
            let lb_dim = mc.min(m - l0);

            let a_il = &a[i0 + l0 * lda ..];
            let b_l  = &mut b[l0 ..];

            sgemm_tn (
                lb_dim,
                n,
                ib_dim,
                -1.0,
                a_il,
                lda,
                x_i,
                ldx,
                1.0,
                b_l,
                ldb,
            );

            l0 += mc;
        }
    }
} 

pub(crate) fn strusm_right_trans(
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

    let mut bj = nb;
    while bj > 0 {
        bj -= 1;

        let j0 = bj * nc;
        let jb_dim = nc.min(n - j0);

        let a_ii = &a[j0 + j0 * lda ..];

        let split = j0 * ldb;
        let (b_left, b_j_and_right) = b.split_at_mut(split);
        let (b_j, _b_right_rest) = b_j_and_right.split_at_mut(jb_dim * ldb);

        backward_sub_panel_right_upper_t (
            m,
            jb_dim,
            unit_diag,
            a_ii,
            lda,
            b_j,
            ldb,
        );

        let x_j: &[f32] = b_j;

        let mut lb = 0;
        while lb < bj {
            let l0 = lb * nc;
            let lb_dim = nc.min(n - l0);

            let u_lj = &a[l0 + j0 * lda ..];
            let b_l  = &mut b_left[l0 * ldb .. (l0 + lb_dim) * ldb];

            sgemm_nt (
                m,        
                lb_dim,   
                jb_dim,   
                -1.0,
                x_j,
                ldb,
                u_lj,
                lda,
                1.0,
                b_l,
                ldb,
            );

            lb += 1;
        }
    }
}

