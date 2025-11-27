use crate::level1::sdot;
use crate::fortran::helpers::ptr_to_vec_ref;

/// LP64 [i32] index unsafe wrapper for [sdot] routine 
///
/// Arguments: 
/// * `n`: [i32]: logical length of vectors 
/// * `x`: *const [f32]: ptr to start of `x` vector 
/// * `incx`: [i32]: stride of `x` vector 
/// * `y`: *const [f32]: ptr to start of `y` vector 
/// * `incy`: [i32]: stride of `y` vector 
///
/// Returns: 
/// [f32] dot product of `x` and `y` 
#[inline] 
pub unsafe fn sdot_lp64( 
    n: i32, 
    x: *const f32, 
    incx: i32, 
    y: *const f32, 
    incy: i32, 
) -> f32 { 
    unsafe { 
        let xview = ptr_to_vec_ref(n, x, incx);
        let yview = ptr_to_vec_ref(n, y, incy); 
        sdot(xview, yview) 
    }
}

/// ILP64 [i64] index unsafe wrapper for [sdot] routine 
///
/// Arguments: 
/// * `n`: [i64]: logical length of vectors 
/// * `x`: *const [f32]: ptr to start of `x` vector 
/// * `incx`: [i64]: stride of `x` vector 
/// * `y`: *const [f32]: ptr to start of `y` vector 
/// * `incy`: [i64]: stride of `y` vector 
///
/// Returns: 
/// [f32] dot product of `x` and `y` 
#[inline] 
pub unsafe fn sdot_ilp64( 
    n: i64, 
    x: *const f32, 
    incx: i64, 
    y: *const f32, 
    incy: i64, 
) -> f32 { 
    unsafe { 
        let xview = ptr_to_vec_ref(n, x, incx);
        let yview = ptr_to_vec_ref(n, y, incy); 
        sdot(xview, yview) 
    }
}

