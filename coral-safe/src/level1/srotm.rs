//! BLAS Level 1 `?ROTM` routine in single precision. 
//!
//! \\[
//! \hat{G}\_{\text{modified}} \begin{pmatrix} x \\\\ y \end{pmatrix}
//! \\]
//!
//! # Author 
//! Deval Deliwala 


use core::simd::Simd; 
use std::simd::StdFloat; 
use std::{slice::IterMut, iter::{StepBy, Take}};
use crate::types::VectorMut; 
use crate::debug_assert_n_eq;

const LANES: usize = 8;
type Vf32 = Simd<f32, LANES>;

/// Organize the algebra for each flag into
/// one helper function for simd 
#[inline] 
fn apply_givens_simd ( 
    xs: &mut [f32], 
    ys: &mut [f32], 
    simd_op: impl Fn(Vf32, Vf32) -> (Vf32, Vf32), // operation on simd vectors 
    scal_op: impl Fn(f32, f32) -> (f32, f32)      // operation on scalar f32s
) { 
    let (xchunks, xtail) = xs.as_chunks_mut::<LANES>(); 
    let (ychunks, ytail) = ys.as_chunks_mut::<LANES>(); 

    // simd chunk
    for (xchunk, ychunk) in xchunks.iter_mut().zip(ychunks.iter_mut()) { 
       let xorig = Simd::<f32, LANES>::from_array(*xchunk); 
       let yorig = Simd::<f32, LANES>::from_array(*ychunk);

       let (xnew, ynew) = simd_op(xorig, yorig);

       *xchunk = xnew.to_array(); 
       *ychunk = ynew.to_array(); 
    }

    // scalar tail
    for (xt, yt) in xtail.iter_mut().zip(ytail.iter_mut()) { 
        let xorig = *xt; 
        let yorig = *yt; 

        let (xnew, ynew) = scal_op(xorig, yorig); 

        *xt = xnew; 
        *yt = ynew;
    }
}

/// Organize the algebra for each flag into 
/// one helper function for scalar 
#[inline] 
fn apply_givens_scal (
    xs_iterator: Take<StepBy<IterMut<f32>>>, 
    ys_iterator: Take<StepBy<IterMut<f32>>>,
    scal_op: impl Fn(f32, f32) -> (f32, f32)
) {
    for (x, y) in xs_iterator.zip(ys_iterator) { 
        let xorig = *x; 
        let yorig = *y; 

        let (xnew, ynew) = scal_op(xorig, yorig);

        *x = xnew; 
        *y = ynew; 
    }
}


/// Updates vectors `x` and `y` using modified Givens transformation 
/// based on given `param` `[f32; 5]`. The form of the transformation 
/// depends on `param[0]`, or the "flag" as follows: 
///
/// `-2.0` - Identity (no op)
/// `-1.0` - General 2x2 matrix `h11, h21, h12, h22` (param[1..5] col major)
/// `0.0`  - Simplified form with implicit ones on diagonal 
/// `+1.0` - Alternate simplified form with fixed off-diagonal ±1s. 
///
/// Arguments: 
/// - `x`: [`VectorMut`] 
/// - `y`: [`VectorMut`] 
/// - `param`: `&[f32; 5]` - [`flag, h11, h21, h12, h22`] that define the modified rotation
///
/// Returns: 
/// Nothing. `x.data` and `y.data` are overwritten. 
#[inline] 
pub fn srotm ( 
    mut x: VectorMut<'_, f32>, 
    mut y: VectorMut<'_, f32>, 
    param: &[f32; 5]
) { 
    debug_assert_n_eq!(x, y);

    let n = x.n(); 
    let incx = x.stride(); 
    let incy = y.stride(); 
    let flag = param[0];
    debug_assert!(matches!(flag, -2.0 | -1.0 | 0.0 | 1.0)); 

    // H stored colmajor
    let h11  = param[1]; 
    let h21  = param[2]; 
    let h12  = param[3]; 
    let h22  = param[4]; 

    // identity 
    if flag == -2.0 { 
        return; 
    } 

    // fast path 
    if let (Some(xs), Some(ys)) = (x.contiguous_slice_mut(), y.contiguous_slice_mut()) { 
        let h11v = Simd::<f32, LANES>::splat(h11);
        let h21v = Simd::<f32, LANES>::splat(h21);
        let h12v = Simd::<f32, LANES>::splat(h12);
        let h22v = Simd::<f32, LANES>::splat(h22);

        match flag {
            // flag = -1.0 
            // x' = [ h11 h12 ] x 
            // y'   [ h21 h22 ] y 
            -1.0 => { 
                apply_givens_simd ( 
                    xs, 
                    ys, 
                    |x, y| { 
                        let xnew = x.mul_add(h11v, h12v * y); 
                        let ynew = x.mul_add(h21v, h22v * y);
                        (xnew, ynew)
                    }, 
                    |x, y| { 
                        let xnew = x.mul_add(h11, h12 * y); 
                        let ynew = x.mul_add(h21, h22 * y); 
                        (xnew, ynew)
                    }
                )
            },

            // flag = 0.0 
            // x' = [ 1.0 h12 ] x 
            // y'   [ h21 1.0 ] y
            0.0 => { 
                apply_givens_simd ( 
                    xs, 
                    ys, 
                    |x, y| { 
                        let xnew = y.mul_add(h12v, x); 
                        let ynew = x.mul_add(h21v, y); 
                        (xnew, ynew)
                    }, 
                    |x, y| { 
                        let xnew = y.mul_add(h12, x); 
                        let ynew = x.mul_add(h21, y);
                        (xnew, ynew)
                    }
                )
            }, 

            // flag = 1.0 
            // x' = [ h11  1.0 ] x
            // y' = [ -1.0 h22 ] y
            1.0 => { 
                apply_givens_simd ( 
                    xs, 
                    ys, 
                    |x, y| { 
                        let xnew = x.mul_add(h11v,  y); 
                        let ynew = y.mul_add(h22v, -x);
                        (xnew, ynew)
                    }, 
                    |x, y| { 
                        let xnew = x.mul_add(h11,  y); 
                        let ynew = y.mul_add(h22, -x);
                        (xnew, ynew)
                    }
                )
            }

            _ => { unreachable!() }
        }

        return;
    }
    
    // slow path 
    let ix = x.offset(); 
    let iy = y.offset(); 
    
    let xs = x.as_slice_mut(); 
    let ys = y.as_slice_mut();

    match flag { 
        -1.0 => { 
            let xs_iterator = xs[ix..].iter_mut().step_by(incx).take(n); 
            let ys_iterator = ys[iy..].iter_mut().step_by(incy).take(n);
            apply_givens_scal ( 
                xs_iterator, 
                ys_iterator,
                |x, y| { 
                    let xnew = h11 * x + h12 * y; 
                    let ynew = h21 * x + h22 * y; 
                    (xnew, ynew) 
                }
            )
        }, 

        0.0 => { 
            let xs_iterator = xs[ix..].iter_mut().step_by(incx).take(n); 
            let ys_iterator = ys[iy..].iter_mut().step_by(incy).take(n);
            apply_givens_scal ( 
                xs_iterator, 
                ys_iterator, 
                |x, y| { 
                    let xnew = x + h12 * y; 
                    let ynew = y + h21 * x; 
                    (xnew, ynew) 
                }
            )
        }, 

        1.0 => { 
            let xs_iterator = xs[ix..].iter_mut().step_by(incx).take(n); 
            let ys_iterator = ys[iy..].iter_mut().step_by(incy).take(n);
            apply_givens_scal (
                xs_iterator, 
                ys_iterator,
                |x, y| { 
                    let xnew = y + h11 * x; 
                    let ynew = -x + h22 * y; 
                    (xnew, ynew)
                }
            )
        }, 

        _ => { unreachable!() }
    }
}
