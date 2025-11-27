use crate::level2::sgemv;
use crate::types::CoralTranspose;
use crate::fortran::helpers::{ptr_to_mat_ref, ptr_to_vec_ref, ptr_to_vec_mut};

/// LP64 [i32] index unsafe wrapper for [sgemv] routine 
///
/// Arguments: 
/// * `op`: [CoralTranspose]: op applied to `a` matrix 
/// * `m`: [i32]: number of rows of `a` matrix 
/// * `n`: [i32]: number of columns of `a` matrix 
/// * `alpha`: [f32]: scalar multiplier for `A x` 
/// * `a`: *const [f32]: ptr to start of `a` matrix 
/// * `lda`: [i32]: leading dimension of `a` matrix 
/// * `x`: *const [f32]: ptr to start of `x` vector 
/// * `incx`: [i32]: stride of `x` vector 
/// * `beta`: [f32]: scalar multiplier for `y` 
/// * `y`: *mut [f32]: ptr to start of `y` vector 
/// * `incy`: [i32]: stride of `y` vector 
///
/// Returns: 
/// Nothing. the contents of `y` are updated in place. 
#[inline]
pub unsafe fn sgemv_lp64(
    op: CoralTranspose,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    unsafe {
        let xview = ptr_to_vec_ref(n, x, incx);
        let yview = ptr_to_vec_mut(m, y, incy);
        let aview = ptr_to_mat_ref(m, n, a, lda);

        sgemv(op, alpha, beta, aview, xview, yview);
    }
}

/// ILP64 [i64] index unsafe wrapper for [sgemv] routine 
///
/// Arguments: 
/// * `op`: [CoralTranspose]: op applied to `a` matrix 
/// * `m`: [i64]: number of rows of `a` matrix 
/// * `n`: [i64]: number of columns of `a` matrix 
/// * `alpha`: [f32]: scalar multiplier for `A x` 
/// * `a`: *const [f32]: ptr to start of `a` matrix 
/// * `lda`: [i64]: leading dimension of `a` matrix 
/// * `x`: *const [f32]: ptr to start of `x` vector 
/// * `incx`: [i64]: stride of `x` vector 
/// * `beta`: [f32]: scalar multiplier for `y` 
/// * `y`: *mut [f32]: ptr to start of `y` vector 
/// * `incy`: [i64]: stride of `y` vector 
///
/// Returns: 
/// Nothing. the contents of `y` are updated in place. 
#[inline]
pub unsafe fn sgemv_ilp64(
    op: CoralTranspose,
    m: i64,
    n: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    x: *const f32,
    incx: i64,
    beta: f32,
    y: *mut f32,
    incy: i64,
) {
    unsafe {
        let xview = ptr_to_vec_ref(n, x, incx);
        let yview = ptr_to_vec_mut(m, y, incy);
        let aview = ptr_to_mat_ref(m, n, a, lda);

        sgemv(op, alpha, beta, aview, xview, yview);
    }
}
