use crate::level2::ssymv;
use crate::fortran::helpers::{ptr_to_vec_ref, ptr_to_vec_mut, ptr_to_mat_ref};
use crate::types::CoralTriangular;

/// LP64 [i32] index unsafe wrapper for [ssymv] routine 
///
/// Arguments: 
/// * `uplo`: [CoralTriangular]: which triangle of `a` is referenced 
/// * `n`: [i32]: order of `a` 
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
pub unsafe fn ssymv_lp64( 
    uplo: CoralTriangular, 
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
        let yview = ptr_to_vec_mut(n, y, incy); 
        let aview = ptr_to_mat_ref(n, n, a, lda); 

        ssymv(uplo, alpha, beta, aview, xview, yview); 
    }
}

/// ILP64 [i64] index unsafe wrapper for [ssymv] routine 
///
/// Arguments: 
/// * `uplo`: [CoralTriangular]: which triangle of `a` is referenced 
/// * `n`: [i64]: order of `a` 
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
pub unsafe fn ssymv_ilp64( 
    uplo: CoralTriangular, 
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
        let yview = ptr_to_vec_mut(n, y, incy); 
        let aview = ptr_to_mat_ref(n, n, a, lda); 

        ssymv(uplo, alpha, beta, aview, xview, yview); 
    }
}

