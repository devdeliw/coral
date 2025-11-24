//! `SCAL`. Scales a single precision vector by a real scalar. 
//!
//! \\[
//! x := \alpha  x
//! \\]
//!
//! This function implements the BLAS [`sscal`] routine, multiplying each element 
//! of the input vector $x$ by the scalar $\alpha$ over $n$ entries with a specified stride.
//!
//! # Author 
//! Deval Deliwala

use crate::level1::assert_length_helpers::required_len_ok; 

/// sscal 
///
/// # Arguments 
/// - `n`     (usize)      : Number of elements to scale. 
/// - `alpha` (f32)        : Scalar multiplier. 
/// - `x`     (&mut [f32]) : Input/output slice containing vector elements. 
/// - `incx`  (usize)      : Stride between consecutive elements of $x$. 
///
/// # Returns 
/// - Nothing. The contents of $x$ are updated in place.
#[inline(always)]
pub fn sscal(
    n       : usize,
    alpha   : f32, 
    x       : &mut [f32],
    incx    : usize
) {
    // quick return
    if n == 0 || incx == 0 { return; }

    debug_assert!(required_len_ok(x.len(), n, incx), "x too short for n/incx");

    // fast path 
    if incx == 1 {
        unsafe {
            let mut p = x.as_mut_ptr();
            let mut i = 0;

            while i < n {
                *p = *p * alpha;
                p  = p.add(1);
                i += 1;
            }
        }
    } else {
        // non unit stride
        unsafe {
            let mut p = x.as_mut_ptr();
            let mut i = 0;

            while i < n {
                *p = *p * alpha;
                p  = p.add(incx);
                i += 1;
            }
        }
    }
}
