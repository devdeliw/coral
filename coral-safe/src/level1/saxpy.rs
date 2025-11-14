//! BLAS Level 1 `?AXPY` routine in single precision. 
//!
//! \\[ 
//! y \leftarrow \alpha x + y
//! \\]
//!
//! # Author 
//! Deval Deliwala


use std::simd::Simd;
use crate::debug_assert_n_eq; 
use crate::types::{VectorRef, VectorMut};


/// Updates [`VectorMut`] `y` by adding `alpha` * `x` [`VectorRef`] 
///
/// Arguments: 
/// - `alpha` f32 : scalar multiplier for `x` 
/// - `x`: [`VectorRef`] over `f32` 
/// - `y`: [`VectorMut`] over `f32`
///
/// Returns: 
/// Nothing. `y.data` is overwritten. 
#[inline] 
pub fn saxpy ( 
    alpha : f32, 
    x     : VectorRef<'_, f32>, 
    mut y : VectorMut<'_, f32>, 
) { 
    debug_assert_n_eq!(x, y);

    let n = x.n(); 
    let incx = x.stride(); 
    let incy = y.stride();
    
    if n == 0 || alpha == 0.0 { 
        return; 
    }
 
    // fast path 
    if let (Some(xs), Some(ys)) = (x.contiguous_slice(), y.contiguous_slice_mut()) { 
        const LANES: usize = 8;
        let a = Simd::<f32, LANES>::splat(alpha); 

        let (xv, xr) = xs.as_chunks::<LANES>();
        let (yv, yr) = ys.as_chunks_mut::<LANES>(); 

        for (xc, yc) in xv.iter().zip(yv.iter_mut()) { 
            let xvec = Simd::from_array(*xc); 
            let yvec = Simd::from_array(*yc);

            // no fma
            let out  = a * xvec + yvec; 

            *yc = out.to_array();
        }

        // scalar remainder tail
        for (xt, yt) in xr.iter().zip(yr.iter_mut()) { 
            *yt += alpha * *xt; 
        }
    } else {
        // scalar fallback
        let ix = x.offset(); 
        let iy = y.offset(); 
        
        let xs = x.as_slice(); 
        let ys = y.as_slice_mut(); 

        let xs_it = xs[ix..].iter().step_by(incx).take(n);
        let ys_it = ys[iy..].iter_mut().step_by(incy).take(n);

        for (&xv, yv) in xs_it.zip(ys_it) {
            *yv += alpha * xv; 
        }
    }
}
