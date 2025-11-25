use crate::level1::sdot; 
use crate::fortran::wrappers::ptr_to_view;

/// unsafe wrapper for [sdot] routine 
#[inline] 
pub unsafe fn sdot_f77 ( 
    n: i32, 
    x: *const f32, 
    incx: i32, 
    y: *const f32, 
    incy: i32, 
) -> f32 { unsafe { 
    let xview = ptr_to_view(n, x, incx);
    let yview = ptr_to_view(n, y, incy); 
    sdot(xview, yview) 
}}


