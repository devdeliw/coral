use crate::level1::snrm2; 
use crate::fortran::wrappers::ptr_to_view;

/// unsafe wrapper for [snrm2] routine 
#[inline] 
pub unsafe fn snrm2_f77 ( 
    n: i32, 
    x: *const f32, 
    incx: i32, 
) -> f32 { unsafe { 
    let xview = ptr_to_view(n, x, incx);
    snrm2(xview) as f32
}}


