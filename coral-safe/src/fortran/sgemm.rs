use crate::level3::sgemm;
use crate::types::CoralTranspose;
use crate::fortran::helpers::{ptr_to_mat_ref, ptr_to_mat_mut};

/// LP64 [i32] index unsafe wrapper for [sgemm] routine 
///
/// Arguments: 
/// * `op_a`: [CoralTranspose]: op applied to `a` matrix 
/// * `op_b`: [CoralTranspose]: op applied to `b` matrix 
/// * `m`: [i32]: number of rows of `c` matrix 
/// * `n`: [i32]: number of columns of `c` matrix 
/// * `k`: [i32]: shared inner dimension 
/// * `alpha`: [f32]: scalar multiplier for `A B` 
/// * `a`: *const [f32]: ptr to start of `a` matrix 
/// * `lda`: [i32]: leading dimension of `a` matrix 
/// * `b`: *const [f32]: ptr to start of `b` matrix 
/// * `ldb`: [i32]: leading dimension of `b` matrix 
/// * `beta`: [f32]: scalar multiplier for `c` 
/// * `c`: *mut [f32]: ptr to start of `c` matrix 
/// * `ldc`: [i32]: leading dimension of `c` matrix 
///
/// Returns: 
/// Nothing. the contents of `c` are updated in place. 
#[inline]
pub unsafe fn sgemm_lp64(
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
) {
    unsafe {
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
    }
}

/// ILP64 [i64] index unsafe wrapper for [sgemm] routine 
///
/// Arguments: 
/// * `op_a`: [CoralTranspose]: op applied to `a` matrix 
/// * `op_b`: [CoralTranspose]: op applied to `b` matrix 
/// * `m`: [i64]: number of rows of `c` matrix 
/// * `n`: [i64]: number of columns of `c` matrix 
/// * `k`: [i64]: shared inner dimension 
/// * `alpha`: [f32]: scalar multiplier for `A B` 
/// * `a`: *const [f32]: ptr to start of `a` matrix 
/// * `lda`: [i64]: leading dimension of `a` matrix 
/// * `b`: *const [f32]: ptr to start of `b` matrix 
/// * `ldb`: [i64]: leading dimension of `b` matrix 
/// * `beta`: [f32]: scalar multiplier for `c` 
/// * `c`: *mut [f32]: ptr to start of `c` matrix 
/// * `ldc`: [i64]: leading dimension of `c` matrix 
///
/// Returns: 
/// Nothing. the contents of `c` are updated in place. 
#[inline]
pub unsafe fn sgemm_ilp64(
    op_a: CoralTranspose,
    op_b: CoralTranspose,
    m: i64,
    n: i64,
    k: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    b: *const f32,
    ldb: i64,
    beta: f32,
    c: *mut f32,
    ldc: i64,
) {
    unsafe {
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
    }
}

