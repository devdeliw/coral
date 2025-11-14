//! BLAS Level 1 `?COPY` routine in single precision. 
//!
//! \\[ 
//! y_i = x_i \forall i \in \mathbb{Z}_n
//! \\]
//!
//! # Author 
//! Deval Deliwala


use crate::types::{VectorRef, VectorMut}; 
use crate::debug_assert_n_eq; 


/// Copys logical elements from `x` [`VectorRef`] into output 
/// `y` [`VectorMut`]. 
///
/// Arguments: 
/// - `x`: [`VectorRef`] over `f32`
/// - `y`: [`VectorMut`] over `f32`
///
/// Returns: 
/// Nothing. `y.data` is ovewritten. 
#[inline] 
pub fn scopy ( 
    x: VectorRef<'_, f32>, 
    mut y: VectorMut<'_, f32>, 
) { 
    debug_assert_n_eq!(x, y); 

    let n = x.n(); 
    let incx = x.stride(); 
    let incy = y.stride();

    if n == 0 { 
        return;
    }
    
    // fast path
    if let (Some(xs), Some(ys)) = (x.contiguous_slice(), y.contiguous_slice_mut()) { 
        for (&xv, yv) in xs.iter().zip(ys.iter_mut()) { 
            *yv = xv;
        }
    } else { 
        // slow path
        let ix = x.offset(); 
        let iy = y.offset(); 

        let xs = x.as_slice(); 
        let ys = y.as_slice_mut(); 

        let xs_it = xs[ix..].iter().step_by(incx).take(n); 
        let ys_it = ys[iy..].iter_mut().step_by(incy).take(n); 

        for (&xv, yv) in xs_it.zip(ys_it) { 
            *yv = xv; 
        }
    }
}   
