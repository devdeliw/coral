use crate::level1::srotm; 
use crate::fortran::wrappers::ptr_to_view_mut; 

/// unsafe wrapper for [srotm] routine 
#[inline] 
pub unsafe fn srotm_f77 ( 
    n: i32, 
    x: *mut f32, 
    incx: i32, 
    y: *mut f32,
    incy: i32, 
    params: *const f32, 
) { unsafe {
    let params_view: &[f32; 5] = &*(params as *const [f32; 5]);
    let xview = ptr_to_view_mut(n, x, incx); 
    let yview = ptr_to_view_mut(n, y, incy);
    srotm(xview, yview, params_view); 
}}
