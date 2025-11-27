use crate::level1::srotg; 

/// I/LP64 unsafe wrapper for [srotg] routine.
/// does not use any [i32]/[i64] indexes.  
///
/// Arguments: 
/// * `a`: *mut [f32]: ptr to `a` scalar 
/// * `b`: *mut [f32]: ptr to `b` scalar 
/// * `c`: *mut [f32]: ptr to `c` scalar 
/// * `s`: *mut [f32]: ptr to `s` scalar 
///
/// Returns: 
/// Nothing. the contents of `a`, `b`, `c`, and `s` are updated in place. 
#[inline] 
pub unsafe fn srotg_f77 ( 
    a: *mut f32, 
    b: *mut f32, 
    c: *mut f32, 
    s: *mut f32, 
) { 
    unsafe { 
        srotg(&mut *a, &mut *b, &mut *c, &mut *s); 
    }
}

