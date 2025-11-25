use crate::level2::ssymv; 
use crate::fortran::wrappers::{ptr_to_view, ptr_to_mat_ref, ptr_to_view_mut}; 
use crate::types::CoralTriangular; 

/// unsafe wrapper for [ssymv] routine 
#[inline] 
pub unsafe fn ssymv_f77 ( 
    uplo: CoralTriangular, 
    n: i32, 
    alpha: f32, 
    a: *const f32, 
    lda: i32, 
    x: *const f32, 
    incx: i32, 
    beta: f32, 
    y: *mut f32, 
    incy: i32 
) { unsafe {
    let xview = ptr_to_view(n, x, incx); 
    let yview = ptr_to_view_mut(n, y, incy); 
    let aview = ptr_to_mat_ref(n, n, a, lda); 

    ssymv(uplo, alpha, beta, aview, xview, yview); 
}}
