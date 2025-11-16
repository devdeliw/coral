//! BLAS Level 1 [`I?AMAX`](https://www.netlib.org/lapack/explore-html/dd/d52/group__iamax.html) 
//! routine in single precision.
//! 
//! \\[ 
//! \text{arg} \max\_{0\leq i < n} \lvert x_i \rvert
//! \\]
//!
//! # Author 
//! Deval Deliwala


use std::simd::Simd; 
use std::simd::num::SimdFloat;
use std::simd::cmp::SimdPartialOrd; 
use crate::types::VectorRef; 


/// Finds the index of the element with maximum absolute value in a
/// single precision [`VectorRef`].
///
/// Arguments: 
/// - `x`: [`VectorRef`] over [`f32`]
///
/// Returns: 
/// - [`usize`] 0-based index of first element with maximum abs value.
#[inline]
pub fn isamax (
    x: VectorRef<'_, f32>
) -> usize {
    let n = x.n();

    if n == 0 {
        return 0;
    }

    // fast path 
    if let Some(xs) = x.contiguous_slice() {
        const LANES: usize = 16;

        let mut max_idx = 0;
        let mut max_val = 0.0;

        let (chunks, tail) = xs.as_chunks::<LANES>();
        for (idx, chunk) in chunks.iter().enumerate() { 
           let vec  = Simd::<f32, LANES>::from_array(*chunk).abs();
           let mask = vec.simd_gt(Simd::splat(max_val));
        
           if mask.any() { 
               for lane in 0..LANES { 
                    let v = vec[lane]; 
                    if v > max_val { 
                        max_val = v; 
                        max_idx = idx * LANES + lane
                    }
               }
           }
        }


        let simd_len = chunks.len() * LANES; 
        for (i, &v) in tail.iter().enumerate() {
            let val = v.abs();
            if val > max_val {
                max_val = val;
                max_idx = simd_len + i;
            }
        }

        return max_idx;
    }

    // slow path 
    let incx = x.stride(); 
    let ix   = x.offset();
    let xs   = x.as_slice();

    let mut max_idx = 0; 
    let mut max_val = 0.0;

    let xs_iterator = xs[ix..]
        .iter()
        .step_by(incx)
        .take(n); 

    for (idx, v) in  xs_iterator.enumerate() { 
        let v = v.abs(); 
        if v > max_val { 
            max_idx = idx; 
            max_val = v; 
        }
    }

    max_idx
}

