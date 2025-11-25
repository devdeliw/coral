use crate::level3::sgemm; 
use crate::types::CoralTranspose; 
use crate::fortran::wrappers::{ptr_to_mat_mut, ptr_to_mat_ref}; 

/// unsafe wrapper for [sgemm] routine 
#[inline]
pub unsafe fn sgemm_f77(
    op_a: CoralTranspose,
    op_b: CoralTranspose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) { unsafe { 
    let (a_rows, a_cols) = match op_a {
        CoralTranspose::NoTrans => (m, k),
        CoralTranspose::Trans   => (k, m),
    };

    let (b_rows, b_cols) = match op_b {
        CoralTranspose::NoTrans => (k, n),
        CoralTranspose::Trans   => (n, k),
    };

    let (c_rows, c_cols) = (m, n);

    let aview = ptr_to_mat_ref(a_rows, a_cols, a, lda);
    let bview = ptr_to_mat_ref(b_rows, b_cols, b, ldb);
    let cview = ptr_to_mat_mut(c_rows, c_cols, c, ldc);

    sgemm(op_a, op_b, alpha, beta, aview, bview, cview);
}}
