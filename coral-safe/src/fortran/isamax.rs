use crate::level1::isamax; 
use crate::fortran::helpers::ptr_to_vec_ref; 

/// LP64 [i32] index unsafe wrapper for [isamax] routine 
///
/// Arguments: 
/// * `n`: [i32]: logical length of vector 
/// * `x`: *const [f32]: ptr to start of vector 
/// * `incx`: [i32]: stride of vector 
///
/// Returns: 
/// 0-based [i32] index 
#[inline] 
pub unsafe fn isamax_lp64 ( 
    n: i32, 
    x: *const f32, 
    incx: i32 
) -> i32 { unsafe { 
    let xview = ptr_to_vec_ref(n, x, incx);
    isamax(xview) as i32 
}}

/// ILP64 [i64] index unsafe wrapper for [isamax] routine 
///
/// Arguments: 
/// * `n`: [i64]: logical length of vector 
/// * `x`: *const [f32]: ptr to start of vector 
/// * `incx`: [i64]: stride of vector 
///
/// Returns: 
/// 0-based [i64] index
#[inline] 
pub unsafe fn isamax_ilp64 ( 
    n: i64, 
    x: *const f32, 
    incx: i64 
) -> i64 { unsafe { 
    let xview = ptr_to_vec_ref(n, x, incx);
    isamax(xview) as i64
}}


