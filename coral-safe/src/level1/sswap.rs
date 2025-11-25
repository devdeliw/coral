//! Level 1 [`?SWAP`](https://www.netlib.org/lapack/explore-html/d7/d51/group__swap.html)
//! routine in single precision 
//!
//! \\[
//! x\_i \\; \leftrightarrow \\; y\_i
//! \\]
//!
//! # Author 
//! Deval Deliwala 


use crate::debug_assert_n_eq; 
use crate::types::VectorMut; 


/// Exchanges logical elements of two input [VectorMut]'s data. 
///
/// Arguments:
/// * `x`: [VectorMut] - over [f32] 
/// * `y`: [VectorMut] - over [f32]
///
/// Returns: 
/// - Nothing. `x` and `y` are swapped in place.
pub fn sswap ( 
    mut x: VectorMut<'_, f32>, 
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
    if let (Some(xs), Some(ys)) = (x.contiguous_slice_mut(), y.contiguous_slice_mut()) { 
        for (xv, yv) in xs.iter_mut().zip(ys.iter_mut()) { 
            core::mem::swap(xv, yv);
        }
    } else {
        // slow path 
        let ix = x.offset(); 
        let iy = y.offset(); 

        let xs = x.as_slice_mut(); 
        let ys = y.as_slice_mut();

        let xs_it = xs[ix..].iter_mut().step_by(incx).take(n);
        let ys_it = ys[iy..].iter_mut().step_by(incy).take(n); 

        for (xv, yv) in xs_it.zip(ys_it) { 
            core::mem::swap(xv, yv);
        }
    }
}

