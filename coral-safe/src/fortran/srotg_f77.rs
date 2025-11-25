use crate::level1::srotg; 

/// unsafe wrapper for [srotg] routine 
#[inline] 
pub unsafe fn srotg_77 ( 
    a: *mut f32, 
    b: *mut f32, 
    c: *mut f32, 
    s: *mut f32, 
) { unsafe { 
    srotg(&mut *a, &mut *b, &mut *c, &mut *s); 
}}
