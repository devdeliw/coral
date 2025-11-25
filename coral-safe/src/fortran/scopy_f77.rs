use crate::level1::scopy; 
use crate::fortran::wrappers::{ptr_to_view, ptr_to_view_mut}; 

/// unsafe wrapper for [scopy] routine 
#[inline] 
pub unsafe fn scopy_77 ( 
    n: i32, 
    x: *const f32, 
    incx: i32, 
    y: *mut f32, 
    incy: i32, 
) { unsafe { 
    let xview = ptr_to_view(n, x, incx);
    let yview = ptr_to_view_mut(n, y, incy); 
    scopy(xview, yview); 
}}


