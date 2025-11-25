use crate::level2::ssyr; 
use crate::fortran::wrappers::{ptr_to_mat_mut, ptr_to_view}; 
use crate::types::CoralTriangular; 

/// unsafe wrapper for [ssyr] routine 
#[inline] 
pub unsafe fn sger_f77 ( 
    uplo: CoralTriangular, 
    n: i32, 
    alpha: f32, 
    x: *const f32,
    incx: i32, 
    a: *mut f32, 
    lda: i32, 
) { unsafe { 
    let xview = ptr_to_view(n, x, incx); 
    let aview = ptr_to_mat_mut(n, n, a, lda); 

    ssyr(uplo, alpha, aview, xview); 
}}

