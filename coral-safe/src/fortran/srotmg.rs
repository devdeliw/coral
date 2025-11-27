use crate::level1::srotmg; 

/// I/LP64 unsafe wrapper for [srotmg] routine. 
/// does not use any [i32]/[i64] indexes. 
///
/// Arguments: 
/// * `d1`: *mut [f32]: ptr to `d1` scalar 
/// * `d2`: *mut [f32]: ptr to `d2` scalar 
/// * `x1`: *mut [f32]: ptr to `x1` scalar 
/// * `y1`: [f32]: `y1` scalar 
/// * `param`: *mut [f32]: ptr to parameter array of length 5 
///
/// Returns: 
/// Nothing. the contents of `d1`, `d2`, `x1`, and `param` are updated in place. 
#[inline]
pub unsafe fn srotmg_f77( 
    d1: *mut f32, 
    d2: *mut f32, 
    x1: *mut f32, 
    y1: f32, 
    param: *mut f32, 
) { 
    unsafe { 
        srotmg(
            &mut *d1,
            &mut *d2,
            &mut *x1,
            y1,
            &mut *(param as *mut [f32; 5]),
        )
    }
}

