use crate::level2::sgemv; 
use crate::types::CoralTranspose; 
use crate::fortran::wrappers::{ptr_to_view_mut, ptr_to_mat_ref, ptr_to_view}; 

// unsafe wrapper for [sgemv] routine 
#[inline] 
pub unsafe fn sgemv_f77 ( 
    op: CoralTranspose, 
    m: i32, 
    n: i32, 
    alpha: f32, 
    a: *const f32, 
    lda: i32, 
    x: * const f32, 
    incx: i32, 
    beta: f32, 
    y: *mut f32, 
    incy: i32, 
) { unsafe {
    let xview = ptr_to_view(n, x, incx); 
    let yview = ptr_to_view_mut(m, y, incy); 
    let aview = ptr_to_mat_ref(m, n, a, lda); 

    sgemv(op, alpha, beta, aview, xview, yview); 
}}
