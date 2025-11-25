use crate::level1::sswap; 
use crate::fortran::wrappers::ptr_to_view_mut; 

/// unsafe wrapper for [sswap] routine 
#[inline] 
pub unsafe fn sswap_77 ( 
    n: i32, 
    x: *mut f32, 
    incx: i32, 
    y: *mut f32, 
    incy: i32, 
) { unsafe { 
    let xview = ptr_to_view_mut(n, x, incx);
    let yview = ptr_to_view_mut(n, y, incy); 
    sswap(xview, yview); 
}}


