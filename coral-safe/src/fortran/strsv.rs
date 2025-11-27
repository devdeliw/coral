use crate::level2::strsv; 
use crate::types::{CoralDiagonal, CoralTranspose, CoralTriangular}; 
use crate::fortran::helpers::{ptr_to_vec_mut, ptr_to_mat_ref}; 

/// LP64 [i32] index unsafe wrapper for [strsv] routine
///
/// Arguments: 
/// * `uplo`: [CoralTriangular]: which triangle of `a` is referenced 
/// * `op`: [CoralTranspose]: op applied to `a` 
/// * `diag`: [CoralDiagonal]: whether the diagonal of `a` is unit or non-unit 
/// * `n`: [i32]: order of `a` 
/// * `a`: *const [f32]: ptr to start of `a` 
/// * `lda`: [i32]: leading dimension of `a` 
/// * `x`: *mut [f32]: ptr to start of `x` 
/// * `incx`: [i32]: stride of `x` 
///
/// Returns: 
/// Nothing. the contents of `x` are updated in place. 
#[inline] 
pub unsafe fn strsv_lp64( 
    uplo: CoralTriangular, 
    op: CoralTranspose, 
    diag: CoralDiagonal, 
    n: i32, 
    a: *const f32, 
    lda: i32, 
    x: *mut f32, 
    incx: i32,
) { 
    unsafe { 
        let xview = ptr_to_vec_mut(n, x, incx); 
        let aview = ptr_to_mat_ref(n, n, a, lda); 

        strsv(uplo, op, diag, aview, xview); 
    }
}

/// ILP64 [i64] index unsafe wrapper for [strsv] routine
///
/// Arguments: 
/// * `uplo`: [CoralTriangular]: which triangle of `a` is referenced 
/// * `op`: [CoralTranspose]: op applied to `a` 
/// * `diag`: [CoralDiagonal]: whether the diagonal of `a` is unit or non-unit 
/// * `n`: [i64]: order of `a` 
/// * `a`: *const [f32]: ptr to start of `a` 
/// * `lda`: [i64]: leading dimension of `a` 
/// * `x`: *mut [f32]: ptr to start of `x` 
/// * `incx`: [i64]: stride of `x` 
///
/// Returns: 
/// Nothing. the contents of `x` are updated in place. 
#[inline] 
pub unsafe fn strsv_ilp64( 
    uplo: CoralTriangular, 
    op: CoralTranspose, 
    diag: CoralDiagonal, 
    n: i64, 
    a: *const f32, 
    lda: i64, 
    x: *mut f32, 
    incx: i64,
) { 
    unsafe { 
        let xview = ptr_to_vec_mut(n, x, incx); 
        let aview = ptr_to_mat_ref(n, n, a, lda); 

        strsv(uplo, op, diag, aview, xview); 
    }
}

