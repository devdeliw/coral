use crate::level2::sger; 
use crate::fortran::wrappers::{ptr_to_mat_mut, ptr_to_view}; 

/// unsafe wrapper for [sger] routine 
#[inline] 
pub unsafe fn sger_f77 ( 
    m: i32, 
    n: i32, 
    alpha: f32, 
    x: *const f32,
    incx: i32, 
    y: *const f32, 
    incy: i32, 
    a: *mut f32, 
    lda: i32, 
) { unsafe { 
    let xview = ptr_to_view(m, x, incx); 
    let yview = ptr_to_view(n, y, incy); 
    let aview = ptr_to_mat_mut(m, n, a, lda); 

    sger(alpha, aview, xview, yview); 
}}
