use crate::level1::srot; 
use crate::fortran::wrappers::ptr_to_view_mut; 

/// unsafe wrapper for [srot] routine 
#[inline] 
pub unsafe fn srot_77 ( 
    n: i32, 
    x: *mut f32, 
    incx: i32, 
    y: *mut f32, 
    incy: i32, 
    c: f32, 
    s: f32,
) { unsafe { 
    let xview = ptr_to_view_mut(n, x, incx);
    let yview = ptr_to_view_mut(n, y, incy); 
    srot(xview, yview, c, s); 
}}


