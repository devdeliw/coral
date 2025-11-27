use crate::level1::snrm2;
use crate::fortran::helpers::ptr_to_vec_ref;

/// LP64 [i32] index unsafe wrapper for [snrm2] routine 
///
/// Arguments: 
/// * `n`: [i32]: logical length of vector 
/// * `x`: *const [f32]: ptr to start of vector 
/// * `incx`: [i32]: stride of vector 
///
/// Returns: 
/// [f32] 2-norm of vector 
#[inline] 
pub unsafe fn snrm2_lp64( 
    n: i32, 
    x: *const f32, 
    incx: i32, 
) -> f32 { 
    unsafe { 
        let xview = ptr_to_vec_ref(n, x, incx);
        snrm2(xview) as f32
    }
}

/// ILP64 [i64] index unsafe wrapper for [snrm2] routine 
///
/// Arguments: 
/// * `n`: [i64]: logical length of vector 
/// * `x`: *const [f32]: ptr to start of vector 
/// * `incx`: [i64]: stride of vector 
///
/// Returns: 
/// [f32] 2-norm of vector 
#[inline] 
pub unsafe fn snrm2_ilp64( 
    n: i64, 
    x: *const f32, 
    incx: i64, 
) -> f32 { 
    unsafe { 
        let xview = ptr_to_vec_ref(n, x, incx);
        snrm2(xview) as f32
    }
}
