//! Level 1 [`?SCAL`](https://www.netlib.org/lapack//explore-html/d2/de8/group__scal.html)
//! routine in single precision. 
//!
//! \\[ 
//! x \leftarrow \alpha x 
//! \\]
//!
//! # Author 
//! Deval Deliwala


use std::simd::Simd; 
use crate::types::VectorMut; 


/// Updates [VectorMut] `x` in place via `x *= alpha`
///
/// Arguments: 
/// * `alpha`: f32 - scalar multiplier for `x` 
/// * `x`: [VectorMut] - struct over [f32]. 
///
/// Returns: 
/// Nothing. `x.data` is overwritten. 
#[inline] 
pub fn sscal ( 
    alpha: f32, 
    mut x: VectorMut<'_, f32>, 
) { 
    let n = x.n(); 
    let incx = x.stride(); 

    if n == 0 { 
        return; 
    }

    // fast path 
    if let Some(xs) = x.contiguous_slice_mut() { 
        const LANES: usize = 32; 
        
        let alpha_vec = Simd::<f32, LANES>::splat(alpha); 
        let  (chunks, tail) = xs.as_chunks_mut::<LANES>(); 
        for xchunk in chunks { 
            let v = Simd::<f32, LANES>::from_array(*xchunk);

            *xchunk = (v * alpha_vec).to_array();
        }

        for xt in tail {
            *xt *= alpha; 
        }

        return;
    }

    // slow path 
    let ix = x.offset(); 
    let xs = x.as_slice_mut(); 

    let xs_it = xs[ix..].iter_mut().step_by(incx).take(n); 

    for xv in xs_it { 
        *xv *= alpha;
    }

}
