use crate::level2::ssyr2; 
use crate::fortran::helpers::{ptr_to_mat_mut, ptr_to_vec_ref}; 
use crate::types::CoralTriangular; 

/// LP64 [i32] index unsafe wrapper for [ssyr2] routine 
///
/// Arguments: 
/// * `uplo`: [CoralTriangular]: which triangle of `a` is referenced 
/// * `n`: [i32]: order of `a` 
/// * `alpha`: [f32]: scalar multiplier 
/// * `x`: *const [f32]: ptr to start of `x` vector 
/// * `incx`: [i32]: stride of `x` vector 
/// * `y`: *const [f32]: ptr to start of `y` vector 
/// * `incy`: [i32]: stride of `y` vector 
/// * `a`: *mut [f32]: ptr to start of `a` matrix 
/// * `lda`: [i32]: leading dimension of `a` matrix 
///
/// Returns: 
/// Nothing. the contents of `a` are updated in place. 
#[inline] 
pub unsafe fn ssyr2_lp64( 
    uplo: CoralTriangular, 
    n: i32, 
    alpha: f32, 
    x: *const f32,
    incx: i32, 
    y: *const f32, 
    incy: i32, 
    a: *mut f32, 
    lda: i32, 
) { 
    unsafe { 
        let xview = ptr_to_vec_ref(n, x, incx);
        let yview = ptr_to_vec_ref(n, y, incy); 
        let aview = ptr_to_mat_mut(n, n, a, lda); 

        ssyr2(uplo, alpha, aview, xview, yview); 
    }
}

/// ILP64 [i64] index unsafe wrapper for [ssyr2] routine 
///
/// Arguments: 
/// * `uplo`: [CoralTriangular]: which triangle of `a` is referenced 
/// * `n`: [i64]: order of `a` 
/// * `alpha`: [f32]: scalar multiplier 
/// * `x`: *const [f32]: ptr to start of `x` vector 
/// * `incx`: [i64]: stride of `x` vector 
/// * `y`: *const [f32]: ptr to start of `y` vector 
/// * `incy`: [i64]: stride of `y` vector 
/// * `a`: *mut [f32]: ptr to start of `a` matrix 
/// * `lda`: [i64]: leading dimension of `a` matrix 
///
/// Returns: 
/// Nothing. the contents of `a` are updated in place. 
#[inline] 
pub unsafe fn ssyr2_ilp64( 
    uplo: CoralTriangular, 
    n: i64, 
    alpha: f32, 
    x: *const f32,
    incx: i64, 
    y: *const f32, 
    incy: i64, 
    a: *mut f32, 
    lda: i64, 
) { 
    unsafe { 
        let xview = ptr_to_vec_ref(n, x, incx);
        let yview = ptr_to_vec_ref(n, y, incy); 
        let aview = ptr_to_mat_mut(n, n, a, lda); 

        ssyr2(uplo, alpha, aview, xview, yview); 
    }
}

