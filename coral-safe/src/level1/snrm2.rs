//! BLAS Level 1 [`?NRM2`](https://www.netlib.org/lapack/explore-html/d1/d2a/group__nrm2.html)
//! routine in single precision. 
//!
//! \\[ 
//! \sqrt{\sum\_{i=0}^{n-1} x_i^2}
//! \\]
//!
//! # Author 
//! Deval Deliwala


use std::simd::Simd;
use std::simd::num::SimdFloat;
use crate::types::VectorRef; 


/// Computes the Euclidean norm of a single precision [`VectorRef`]
///
/// Arguments: 
/// `x`: [`VectorRef`] over [`f32`]
///
/// Returns: 
/// - [`f32`] norm of the logical vector elements. 
#[inline] 
pub fn snrm2 ( 
    x: VectorRef<'_, f32>
) -> f32 { 
    let n    = x.n(); 
    let incx = x.stride(); 

    if n == 0 || incx == 0 { 
         return 0.0; 
    }

    let mut sum: f32 = 0.0; 
    if let Some(xs) = x.contiguous_slice() { 
        const LANES: usize = 32; 

        let (chunks, tail) = xs.as_chunks::<LANES>(); 
        for &chunk in chunks { 
            let v = Simd::<f32, LANES>::from_array(chunk); 

            sum += (v * v).reduce_sum();
        }

        // no fma
        for &t in tail { 
            sum += t * t;
        }

        return sum.sqrt(); 
    }

    let ix = x.offset(); 
    let xs = x.as_slice(); 

    for &v in xs[ix..].iter().step_by(incx).take(n) { 
        sum += v * v; 
    }

    sum.sqrt()
}
