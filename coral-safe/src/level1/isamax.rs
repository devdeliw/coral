//! BLAS Level 1 `I?AMAX` routine in single precision.
//! 
//! \\[ 
//! \text{arg} \text{max}\_{0\leq i < n} \lvert x_i \rvert
//! \\]
//!
//! # Author 
//! Deval Deliwala


use std::simd::Simd; 
use std::simd::num::SimdFloat; 
use crate::types::VectorRef; 


/// Finds the index of the element with maximum absolute value in a
/// single precision [`VectorRef`].
///
/// Arguments: 
/// - `x`: [`VectorRef`] over `f32`
///
/// Returns: 
/// - `usize` 0-based index of first element with maximum abs value.
#[inline] 
pub fn isamax ( 
    x: VectorRef<'_, f32>
) -> usize { 
    let n = x.n(); 
    let incx = x.stride(); 

    if n == 0 { 
        return 0; 
    }   

    let mut max_val = 0.0; 

    // fast path
    if let Some(xs) = x.contiguous_slice() { 
        const LANES: usize = 16; 
        let (chunks, tail) = xs.as_chunks::<LANES>();

        for &chunk in chunks.iter() { 
            let chunk_vec = Simd::from_array(chunk).abs();
            let val = chunk_vec.reduce_max(); 
            if val > max_val { 
                max_val = val; 
            }
        }

        for &xt in tail.iter() { 
           let val = xt.abs(); 
           if val > max_val { 
                max_val = val; 
           }
        }

        for (idx, &val) in xs.iter().enumerate() { 
            if val.abs() == max_val { 
                return idx; 
            }
        }
    }

    // slow path 
    let mut max_idx = 0;
    let ix = x.offset(); 
    let xs = x.as_slice();  
    let xs_it = xs[ix..].iter().step_by(incx).take(n); 

    for (idx, &xc) in xs_it.enumerate() { 
        let val = xc.abs(); 

        if val > max_val { 
            max_val = val; 
            max_idx = idx; 
        }
    }
    
    max_idx
}
