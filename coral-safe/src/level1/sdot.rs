use std::simd::Simd;
use std::simd::num::SimdFloat;
use crate::types::VectorRef; 
use crate::debug_assert_n_eq; 

#[inline] 
pub fn sdot ( 
    x: VectorRef<'_, f32>, 
    y: VectorRef<'_, f32>, 
) -> f32 {
    debug_assert_n_eq!(x, y); 

    let n = x.n(); 
    let incx = x.stride(); 
    let incy = y.stride(); 

    if n == 0 { 
        return 0.0;
    }

    // fast path
    if let (Some(xs), Some(ys)) = (x.contiguous_slice(), y.contiguous_slice()) { 
        const LANES: usize = 32; 
        let mut acc = Simd::<f32, LANES>::splat(0.0);

        let (xv, xt) = xs.as_chunks::<LANES>(); 
        let (yv, yt) = ys.as_chunks::<LANES>(); 

        for (&xc, &yc) in xv.iter().zip(yv.iter()) { 
            let xm = Simd::from_array(xc); 
            let ym = Simd::from_array(yc);

            // no fma
            acc += xm * ym; 
        }

        let mut acc_tail = 0.0; 
        for (&xf, &yf) in xt.iter().zip(yt.iter()) { 
            acc_tail += xf * yf; 
        }

        return acc.reduce_sum() + acc_tail;
    }  

    // slow path 
    let mut acc = 0.0; 
    let ix = x.offset(); 
    let iy = y.offset(); 

    let xs = x.as_slice(); 
    let ys = y.as_slice(); 

    let xs_it = xs[ix..].iter().step_by(incx).take(n); 
    let ys_it = ys[iy..].iter().step_by(incy).take(n); 

    for (&xv, &yv) in xs_it.zip(ys_it) { 
        acc += xv * yv; 
    }

    acc
}
