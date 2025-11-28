use crate::level3::sgemm_nn; 
use crate::level3::subs_lower::{ 
    forward_sub_panel_left_lower_n,
    backward_sub_panel_right_lower_n
}; 
use crate::level3::sgemm::{MC, NC};
use crate::fused::sscalf;
use crate::types::MatrixMut;

pub(crate) fn strlsm_left_notrans ( 
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

        let a_ii = &a[i0 + i0 * lda ..]; 
        let b_i  = &mut b[i0 ..]; 

        forward_sub_panel_left_lower_n ( 
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
        let (bi_buf, _rest) = scratch.split_at_mut(panel_len);

        // copy solved panel into scratch buffer 
        for j in 0..n { 
            let src_col = &b[i0 + j * ldb .. i0 + j * ldb + ib_dim]; 
            let dst_col = &mut bi_buf[j * ib_dim .. (j + 1) * ib_dim]; 
            dst_col.copy_from_slice(src_col); 
        }

        let x_i: &[f32] = &bi_buf[..]; 
        let ldb_xi = ib_dim; 

        // rows below 
        let mut l0 = i0 + ib_dim; 
        while l0 < m { 
            let lb_dim = mc.min(m - l0); 

            let a_li = &a[l0 + i0 * lda ..]; 
            let b_l  = &mut b[l0 ..]; 

            sgemm_nn ( 
                lb_dim,   
                n,        
                ib_dim,   
                -1.0,     
                a_li, 
                lda, 
                x_i, 
                ldb_xi, 
                1.0,      
                b_l, 
                ldb, 
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

        let l_ii = &a[j0 + j0 * lda ..];

        let split = j0 * ldb;
        let (b_left, b_j_and_right) = b.split_at_mut(split);
        let (b_j, _b_right_rest) = b_j_and_right.split_at_mut(jb_dim * ldb);

        backward_sub_panel_right_lower_n (
            m,
            jb_dim,
            unit_diag,
            l_ii,
            lda,
            b_j,
            ldb,
        );

        let x_j: &[f32] = b_j;

        let mut lb = 0;
        while lb < bj {
            let k0 = lb * nc;
            let kb_dim = nc.min(n - k0);

            let l_jl = &a[j0 + k0 * lda ..];
            let b_l  = &mut b_left[k0 * ldb .. (k0 + kb_dim) * ldb];

            sgemm_nn (
                m,
                kb_dim,
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
