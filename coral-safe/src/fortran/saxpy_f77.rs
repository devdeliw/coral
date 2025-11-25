use crate::level1::saxpy; 
use crate::fortran::wrappers::{ptr_to_view, ptr_to_view_mut}; 

/// unsafe wrapper for [saxpy] routine 
#[inline] 
pub unsafe fn saxpy_f77 ( 
    n: i32, 
    alpha: f32,
    x: *const f32, 
    incx: i32, 
    y: *mut f32, 
    incy: i32, 
) { unsafe { 
    let xview = ptr_to_view(n, x, incx);
    let yview = ptr_to_view_mut(n, y, incy); 
    saxpy(alpha, xview, yview); 
}}

