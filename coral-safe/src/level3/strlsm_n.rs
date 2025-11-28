use crate::level3::sgemm_nn::sgemm_nn; 
use crate::level3::sgemm::{MC, NC}; 
use crate::fused::sscalf; 
use crate::level3::substitutions::{
    forward_sub_panel_left_lower_n, 
    backward_sub_panel_right_lower_n
}; 
use crate::types::MatrixMut; 

pub(crate) fn strlsm_left_notrans ( 
    m: usize, 
    n: usize, 
    alpha: f32, 
    unit_diag: bool, 
    a: &[f32], 
    lda: usize, 
    b: &mut [f32], 
    ldb: usize  
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

        // solve diagonal block 
        let a_ii = &a[i0 + i0 * lda..]; 
        let b_i = &mut b[i0..]; 

        forward_sub_panel_left_lower_n ( 
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

        // rows below diagonal block 
        let mut l0 = i0 + ib_dim; 
        while l0 < m { 
            let lb_dim = mc.min(m - l0); 

            let a_li = &a[l0 + i0 * lda..]; 
            let b_l = &mut b[l0..]; 
            let b_i = bi_buf.as_slice(); 
            let ldb_bi = ib_dim; 

            sgemm_nn ( 
                lb_dim, 
                n, 
                ib_dim, 
                -1.0, 
                a_li, 
                lda, 
                b_i, 
                ldb_bi, 
                1.0, 
                b_l, 
                ldb
            ); 

            l0 += mc; 
        }
    }
}

pub(crate) fn strlsm_right_notrans (
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

        let l_jj = &a[j0 + j0 * lda..];

        let split = j0 * ldb;
        let (b_left, b_j_and_right) = b.split_at_mut(split);
        let (b_j, _b_right_rest) = b_j_and_right.split_at_mut(jb_dim * ldb);

        backward_sub_panel_right_lower_n (
            m,
            jb_dim,
            unit_diag,
            l_jj,
            lda,
            b_j,
            ldb,
        );

        let x_j: &[f32] = b_j;

        let mut lb = 0;
        while lb < bj {
            let l0 = lb * nc;
            let lb_dim = nc.min(n - l0);

            let l_jl = &a[j0 + l0 * lda..];
            let b_l  = &mut b_left[l0 * ldb .. (l0 + lb_dim) * ldb];

            sgemm_nn(
                m,
                lb_dim,
                jb_dim,
                -1.0,
                x_j,
                ldb,
                l_jl,
                lda,
                1.0,
                b_l,
                ldb,
            );

            lb += 1;
        }
    }
}

