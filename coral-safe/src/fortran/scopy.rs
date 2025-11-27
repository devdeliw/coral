use crate::level1::scopy;
use crate::fortran::helpers::{ptr_to_vec_ref, ptr_to_vec_mut};

/// LP64 [i32] index unsafe wrapper for [scopy] routine 
///
/// Arguments: 
/// * `n`: [i32]: logical length of vector 
/// * `x`: *const [f32]: ptr to start of `x` vector 
/// * `incx`: [i32]: stride of `x` vector 
/// * `y`: *mut [f32]: ptr to start of `y` vector 
/// * `incy`: [i32]: stride of `y` vector 
///
/// Returns: 
/// Nothing. the contents of `y` are updated in place. 
#[inline]
pub unsafe fn scopy_lp64(
    n: i32,
    x: *const f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
) {
    unsafe {
        let xview = ptr_to_vec_ref(n, x, incx);
        let yview = ptr_to_vec_mut(n, y, incy);
        scopy(xview, yview);
    }
}

/// ILP64 [i64] index unsafe wrapper for [scopy] routine 
///
/// Arguments: 
/// * `n`: [i64]: logical length of vector 
/// * `x`: *const [f32]: ptr to start of `x` vector 
/// * `incx`: [i64]: stride of `x` vector 
/// * `y`: *mut [f32]: ptr to start of `y` vector 
/// * `incy`: [i64]: stride of `y` vector 
///
/// Returns: 
/// Nothing. the contents of `y` are updated in place. 
#[inline]
pub unsafe fn scopy_ilp64(
    n: i64,
    x: *const f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
) {
    unsafe {
        let xview = ptr_to_vec_ref(n, x, incx);
        let yview = ptr_to_vec_mut(n, y, incy);
        scopy(xview, yview);
    }
}

