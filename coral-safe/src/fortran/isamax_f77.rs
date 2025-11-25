use crate::level1::isamax; 
use crate::fortran::wrappers::ptr_to_view; 

/// unsafe wrapper for [isamax] routine 
#[inline] 
pub unsafe fn isamax_f77 ( 
    n: i32, 
    x: *const f32, 
    incx: i32 
) -> i32 { unsafe { 
    let xview = ptr_to_view(n, x, incx);
    isamax(xview) as i32 
}}
