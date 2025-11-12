use core::simd::Simd; 
use core::simd::num::SimdFloat;
use crate::types::VectorRef; 

#[inline(always)] 
pub fn sasum ( 
    x: VectorRef<'_, f32>
) -> f32 { 
    let n = x.n(); 
    let incx = x.stride();

    if n == 0 || incx == 0 { 
        return 0.0; 
    }

    if let Some(xs) = x.contiguous_slice() { 
        const LANES: usize = 32; 
        let mut acc = Simd::<f32, LANES>::splat(0.0);

        let (chunks, tail) = xs.as_chunks::<LANES>(); 
        for &chunk in chunks { 
            let v = Simd::<f32, LANES>::from_array(chunk);
            acc += v.abs();
        }

        // horizontal sum of accumulator 
        let mut res: f32 = acc.reduce_sum(); 

        // scalar remainder tail 
        for &t in tail { 
            res += t.abs();
        }

        return res;
    }  

    // scalar fallback 
    let mut res = 0.0;
    let ix  = x.offset(); 
    let xs  = x.as_slice(); 

    for &v in xs[ix..].iter().step_by(incx).take(n) { 
        res += v.abs();
    }

    res
}
