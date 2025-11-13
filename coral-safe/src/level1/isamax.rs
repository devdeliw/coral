use crate::types::VectorRef;
use std::simd::Simd;
use std::simd::num::SimdFloat;

#[inline]
pub fn isamax (
    x: VectorRef<'_, f32>
) -> usize {
    let n    = x.n();
    let incx = x.stride();

    if n == 0 || incx == 0 {
        return 0;
    }

    // fast path
    if let Some(xs) = x.contiguous_slice() {
        const LANES: usize = 32;

        let len = xs.len();
        let mut max_idx = 0;
        let mut max_val = 0.0;

        let mut i = 0;
        let end = len - (len % LANES);

        while i < end {
            let chunk = Simd::<f32, LANES>::from_slice(&xs[i..i + LANES]);
            let abs_chunk = chunk.abs();

            let chunk_array = abs_chunk.to_array();

            let mut local_max = chunk_array[0];
            let mut local_idx = 0;

            // local max 
            for j in 1..LANES {
                if chunk_array[j] > local_max {
                    local_max = chunk_array[j];
                    local_idx = j;
                }
            }

            // global max
            if local_max > max_val {
                max_val = local_max;
                max_idx = i + local_idx;
            }

            i += LANES;
        }

        while i < len {
            let v = xs[i].abs();
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
            i += 1;
        }

        return max_idx;
    }

    // slow path
    let ix = x.offset();
    let xs = x.as_slice();

    let mut max_idx = 0;
    let mut max_val = 0.0;

    let xs_it = xs[ix..].iter().step_by(incx).take(n);

    for (idx, v) in xs_it.enumerate() {
        let v = v.abs();
        if v > max_val {
            max_val = v;
            max_idx = idx;
        }
    }

    max_idx
}

