use crate::level3::sgemm_nn;
use crate::level3::sgemm::{MC, NC};
use crate::fused::sscalf;
use crate::level3::subs_upper::{
    backward_sub_panel_left_upper_n,
    forward_sub_panel_right_upper_n, 
};
use crate::types::MatrixMut;

pub(crate) fn strusm_left_notrans (
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

    let mut bi = mb;
    while bi > 0 {
        bi -= 1;

        let i0 = bi * mc;
        let ib_dim = mc.min(m - i0);

        // solve diagonal 
        let a_ii = &a[i0 + i0 * lda ..];
        let b_i  = &mut b[i0 ..];

        backward_sub_panel_left_upper_n (
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

        let mut bk = 0;
        while bk < bi {
            let k0 = bk * mc;
            let kb_dim = mc.min(m - k0);

            let a_ki = &a[k0 + i0 * lda ..];
            let b_k  = &mut b[k0 ..];

            sgemm_nn (
                kb_dim,   
                n,        
                ib_dim,   
                -1.0,
                a_ki,
                lda,
                x_i,
                ldx,
                1.0,
                b_k,
                ldb,
            );

            bk += 1;
        }
    }
}

pub(crate) fn strusm_right_notrans (
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

        let a_ii = &a[j0 + j0 * lda ..];

        let split = j0 * ldb;
        let (_b_left, b_j_and_right) = b.split_at_mut(split);
        let (b_j, b_right) = b_j_and_right.split_at_mut(jb_dim * ldb);

        forward_sub_panel_right_upper_n (
            m,
            jb_dim,
            unit_diag,
            a_ii,
            lda,
            b_j,
            ldb,
        );

        let jr0 = j0 + jb_dim;
        if jr0 < n {
            let nr = n - jr0;

            let u_jr = &a[j0 + jr0 * lda ..];

            sgemm_nn(
                m,        
                nr,       
                jb_dim,   
                -1.0,
                b_j,      
                ldb,
                u_jr,     
                lda,
                1.0,
                b_right,  
                ldb,
            );
        }
    }
}

