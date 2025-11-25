use crate::level1::srotmg; 

/// unsafe wrapper for [srotmg] routine 
#[inline]
pub unsafe fn srotmg_f77 ( 
    d1: *mut f32, 
    d2: *mut f32, 
    x1: *mut f32, 
    y1: f32, 
    param: *mut f32, 
) { unsafe { 
    srotmg ( 
        &mut *d1, 
        &mut *d2, 
        &mut *x1, 
        y1, 
        &mut *(param as *mut [f32; 5])
    )
}}
