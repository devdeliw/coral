use crate::level1::sasum; 
use crate::fortran::wrappers::ptr_to_view; 

/// unsafe wrapper for [sasum] routine
#[inline] 
pub unsafe fn sasum_f77 ( 
    n: i32, 
    x: *const f32, 
    incx: i32 
) -> f32 { unsafe { 
    let xview = ptr_to_view(n, x, incx);
    sasum(xview)
}}

