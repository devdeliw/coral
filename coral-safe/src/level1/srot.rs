//! Level 1 [`?ROT`](https://www.netlib.org/lapack/explore-html/d1/d45/group__rot.html)
//! routine in single precision
//!
//! \\[ 
//! x \leftarrow c x + s y 
//! \\] 
//! \\[ 
//! y \leftarrow c y - s x 
//! \\] 
//!
//! # Author 
//! Deval Deliwala 


use std::simd::{Simd, StdFloat}; 
use crate::types::VectorMut; 
use crate::debug_assert_n_eq; 


/// Replaces elements in [VectorMut] `x` and `y`
/// with `x := cx + sy` and `y := cy - sx`; i.e. a 
/// 2D Givens rotation. 
///
/// Arguments: 
/// * `x`: [VectorMut] - over [f32] 
/// * `y`: [VectorMut] - over [f32]
/// * `c`: [f32] 
/// * `y`: [f32] 
///
/// Returns: 
/// Nothing. `x` and `y` are overwritten. 
pub fn srot ( 
    mut x: VectorMut<'_, f32>, 
    mut y: VectorMut<'_, f32>, 
    c: f32, 
    s: f32, 
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
        const LANES: usize = 8; 

        let cvec = Simd::<f32, LANES>::splat(c); 
        let svec = Simd::<f32, LANES>::splat(s); 
        let (xchunks, xtail) = xs.as_chunks_mut::<LANES>(); 
        let (ychunks, ytail) = ys.as_chunks_mut::<LANES>();

        for (xchunk, ychunk) in xchunks.iter_mut().zip(ychunks.iter_mut()) { 
            let mut xvec = Simd::from_array(*xchunk); 
            let mut yvec = Simd::from_array(*ychunk); 

            let xorig = xvec; 
            let yorig = yvec;

            // x := cx + sy 
            // y := cy - sx
            xvec = xorig.mul_add(cvec,  yorig * svec); 
            yvec = yorig.mul_add(cvec, -xorig * svec);

            *xchunk = xvec.to_array();
            *ychunk = yvec.to_array(); 
        }

        for (xtail, ytail) in xtail.iter_mut().zip(ytail.iter_mut()) { 
            let xorig = *xtail; 
            let yorig = *ytail; 

            let xnew = xtail.mul_add(c,  yorig * s); 
            let ynew = ytail.mul_add(c, -xorig * s); 

            *xtail = xnew; 
            *ytail = ynew; 
        }
    } else { 
        // slow path 
        let ix = x.offset(); 
        let iy = y.offset(); 
        let xs = x.as_slice_mut(); 
        let ys = y.as_slice_mut(); 

        let xs_it = xs[ix..].iter_mut().step_by(incx).take(n); 
        let ys_it = ys[iy..].iter_mut().step_by(incy).take(n); 

        for (xval, yval) in xs_it.zip(ys_it) { 
           let xorig = *xval; 
           let yorig = *yval;

           let xnew = xval.mul_add(c,  yorig * s); 
           let ynew = yval.mul_add(c, -xorig * s); 

           *xval = xnew; 
           *yval = ynew;
        }
    }
}
