//! BLAS Level 1 [`?ASUM`](https://www.netlib.org/lapack/explore-html/d5/d72/group__asum.html) 
//! routine in single precision.
//!
//! \\[ 
//! \sum\_{i=0}^{n-1} \lvert x_i \rvert
//! \\]
//!
//! # Author 
//! Deval Deliwala 


use std::simd::Simd; 
use std::simd::num::SimdFloat;
use crate::types::VectorRef; 


/// Computes the sum of absolute values of elements in a 
/// single precision [`VectorRef`].
///
/// Arguments: 
/// - `x`: [`VectorRef`] over [`f32`]
///
/// Returns: 
/// - [`f32`] sum of abs values of logical vector elements. 
#[inline] 
pub fn sasum ( 
    x: VectorRef<'_, f32>
) -> f32 { 
    let n = x.n(); 
    let incx = x.stride();

    if n == 0 || incx == 0 { 
        return 0.0; 
    }

    if let Some(xs) = x.contiguous_slice() { 
        const LANES: usize = 32; 
        let mut acc = Simd::<f32, LANES>::splat(0.0);

        let (chunks, tail) = xs.as_chunks::<LANES>(); 
        for &chunk in chunks { 
            let v = Simd::<f32, LANES>::from_array(chunk);
            acc += v.abs();
        }

        // horizontal sum of accumulator 
        let mut res: f32 = acc.reduce_sum(); 

        // scalar remainder tail 
        for &t in tail { 
            res += t.abs();
        }

        return res;
    }  

    // scalar fallback 
    let mut res = 0.0;
    let ix  = x.offset(); 
    let xs  = x.as_slice(); 

    for &v in xs[ix..].iter().step_by(incx).take(n) { 
        res += v.abs();
    }

    res
}
