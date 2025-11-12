use core::simd::Simd;
use crate::types::{VectorRef, VectorMut}; 

#[inline(always)] 
pub fn saxpy ( 
    alpha : f32, 
    x     : VectorRef<'_, f32>, 
    mut y : VectorMut<'_, f32>, 
) { 
    debug_assert!(
        x.n() == y.n(), 
        "number of logical elements must be equal"
    );

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
        let mut ix = x.offset(); 
        let mut iy = y.offset(); 
        
        let xs = x.as_slice(); 
        let ys = y.as_mut_slice(); 

        for _ in 0..n { 
            ys[iy] += alpha * xs[ix]; 

            ix += incx; 
            iy += incy; 
        }
    }
}
