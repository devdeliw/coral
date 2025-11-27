use crate::level1::srotm;
use crate::fortran::helpers::ptr_to_vec_mut;

/// LP64 [i32] index unsafe wrapper for [srotm] routine 
///
/// Arguments: 
/// * `n`: [i32]: logical length of vectors 
/// * `x`: *mut [f32]: ptr to start of `x` vector 
/// * `incx`: [i32]: stride of `x` vector 
/// * `y`: *mut [f32]: ptr to start of `y` vector 
/// * `incy`: [i32]: stride of `y` vector 
/// * `params`: *const [f32]: ptr to parameter array of length 5 
///
/// Returns: 
/// Nothing. the contents of `x` and `y` are updated in place. 
#[inline]
pub unsafe fn srotm_lp64(
    n: i32,
    x: *mut f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
    params: *const f32,
) {
    unsafe {
        let params_view: &[f32; 5] = &*(params as *const [f32; 5]);
        let xview = ptr_to_vec_mut(n, x, incx);
        let yview = ptr_to_vec_mut(n, y, incy);
        srotm(xview, yview, params_view);
    }
}

/// ILP64 [i64] index unsafe wrapper for [srotm] routine 
///
/// Arguments: 
/// * `n`: [i64]: logical length of vectors 
/// * `x`: *mut [f32]: ptr to start of `x` vector 
/// * `incx`: [i64]: stride of `x` vector 
/// * `y`: *mut [f32]: ptr to start of `y` vector 
/// * `incy`: [i64]: stride of `y` vector 
/// * `params`: *const [f32]: ptr to parameter array of length 5 
///
/// Returns: 
/// Nothing. the contents of `x` and `y` are updated in place. 
#[inline]
pub unsafe fn srotm_ilp64(
    n: i64,
    x: *mut f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
    params: *const f32,
) {
    unsafe {
        let params_view: &[f32; 5] = &*(params as *const [f32; 5]);
        let xview = ptr_to_vec_mut(n, x, incx);
        let yview = ptr_to_vec_mut(n, y, incy);
        srotm(xview, yview, params_view);
    }
}

