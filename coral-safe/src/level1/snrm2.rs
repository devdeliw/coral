use core::simd::Simd;
use core::simd::num::SimdFloat;
use crate::types::VectorRef; 

#[inline(always)] 
pub fn snrm2 ( 
    x: VectorRef<'_, f32>
) -> f32 { 
    let n    = x.n(); 
    let incx = x.stride(); 

    if n == 0 || incx == 0 { 
         return 0.0; 
    }

    let mut sum: f32 = 0.0; 
    if let Some(xs) = x.contiguous_slice() { 
        const LANES: usize = 32; 

        let (chunks, tail) = xs.as_chunks::<LANES>(); 
        for &chunk in chunks { 
            let v = Simd::<f32, LANES>::from_array(chunk); 

            sum += (v * v).reduce_sum();
        }

        // scalar remainder tail 
        for &t in tail { 
            sum += t * t;
        }

        return sum.sqrt(); 
    }

    let ix = x.offset(); 
    let xs = x.as_slice(); 

    // let compiler avoid bound checks
    for &v in xs[ix..].iter().step_by(incx).take(n) { 
        sum += v * v; 
    }

    sum.sqrt()
}
