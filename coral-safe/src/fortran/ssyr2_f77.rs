use crate::level2::ssyr2; 
use crate::fortran::wrappers::{ptr_to_mat_mut, ptr_to_view}; 
use crate::types::CoralTriangular; 

/// unsafe wrapper for [ssyr2] routine 
#[inline] 
pub unsafe fn ssyr2_f77 ( 
    uplo: CoralTriangular, 
    n: i32, 
    alpha: f32, 
    x: *const f32,
    incx: i32, 
    y: *const f32, 
    incy: i32, 
    a: *mut f32, 
    lda: i32, 
) { unsafe { 
    let xview = ptr_to_view(n, x, incx);
    let yview = ptr_to_view(n, y, incy); 
    let aview = ptr_to_mat_mut(n, n, a, lda); 

    ssyr2(uplo, alpha, aview, xview, yview); 
}}


