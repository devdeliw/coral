//! Level 1.5 (fused)
//! [`SDOT`](https://www.netlib.org/lapack/explore-html/d1/dcc/group__dot.html)
//! routine. Essentially a transpose SGEMV, accumulating all column dot products in 
//! one sweep, instead of calling [crate::level1::sdot] n_cols times. 
//!
//! \\[ 
//! y \leftarrow y + A^T x
//! \\]
//!
//! # Author 
//! Deval Deliwalia


use std::simd::{Simd, StdFloat}; 
use std::simd::num::SimdFloat;
use crate::types::{MatrixRef, VectorRef, VectorMut}; 
use crate::level1::sdot; 

const MR    : usize = 256; 
const NR    : usize = 4; 
const LANES : usize = 32; 


#[inline] 
fn sdotf_contiguous( 
    n_rows: usize, 
    n_cols: usize, 
    x: &[f32], 
    a: &[f32], 
    lda: usize, 
    y: &mut [f32], 
) { 
    // MR x n_cols
    let mut row_base = 0; 
    while row_base < n_rows { 
        let mr = (n_rows - row_base).min(MR); 
        
        // x panel view to be used
        let x_panel = &x[row_base .. row_base + mr]; 

        // xchunks contains [[f32; 32]; n_chunks]
        let (xchunks, xtail) = x_panel.as_chunks::<LANES>(); 
        let n_chunks = xchunks.len(); 

        if n_chunks > 0 { 
            // LANES x NR 
            let mut col = 0; 
            while col + NR <= n_cols { 

                let mut acc0 = Simd::<f32, LANES>::splat(0.0); 
                let mut acc1 = Simd::<f32, LANES>::splat(0.0); 
                let mut acc2 = Simd::<f32, LANES>::splat(0.0); 
                let mut acc3 = Simd::<f32, LANES>::splat(0.0); 

                let col0 = &a[col * lda + row_base .. col * lda + row_base + mr]; 
                let col1 = &a[(col + 1) * lda + row_base .. (col + 1) * lda + row_base + mr]; 
                let col2 = &a[(col + 2) * lda + row_base .. (col + 2) * lda + row_base + mr]; 
                let col3 = &a[(col + 3) * lda + row_base .. (col + 3) * lda + row_base + mr]; 

                // col0_chunks contains [[f32; 32]; n_chunks] 
                let (col0_chunks, _) = col0.as_chunks::<LANES>();
                let (col1_chunks, _) = col1.as_chunks::<LANES>(); 
                let (col2_chunks, _) = col2.as_chunks::<LANES>(); 
                let (col3_chunks, _) = col3.as_chunks::<LANES>(); 

                // fused FMAs across NR columns at a time 
                // overwrites the first n_rows - n_rows % MR elements of y
                for (chunk_idx, xchunk) in xchunks.iter().enumerate() { 
                    let xv = Simd::<f32, LANES>::from_array(*xchunk);

                    let a0 = Simd::<f32, LANES>::from_array(col0_chunks[chunk_idx]); 
                    let a1 = Simd::<f32, LANES>::from_array(col1_chunks[chunk_idx]); 
                    let a2 = Simd::<f32, LANES>::from_array(col2_chunks[chunk_idx]); 
                    let a3 = Simd::<f32, LANES>::from_array(col3_chunks[chunk_idx]);

                    acc0 = a0.mul_add(xv, acc0); 
                    acc1 = a1.mul_add(xv, acc1); 
                    acc2 = a2.mul_add(xv, acc2); 
                    acc3 = a3.mul_add(xv, acc3); 
                }

                let mut sum0 = acc0.reduce_sum(); 
                let mut sum1 = acc1.reduce_sum(); 
                let mut sum2 = acc2.reduce_sum(); 
                let mut sum3 = acc3.reduce_sum(); 


                if !xtail.is_empty() { 
                    let tail_beg = n_chunks * LANES; 
                    for (r, &xr) in xtail.iter().enumerate() { 
                        let row = tail_beg + r; 

                        sum0 += col0[row] * xr; 
                        sum1 += col1[row] * xr; 
                        sum2 += col2[row] * xr; 
                        sum3 += col3[row] * xr; 
                    }
                }

                y[col] += sum0; 
                y[col + 1] += sum1; 
                y[col + 2] += sum2; 
                y[col + 3] += sum3; 

                col += NR;
            }

            // leftover columns 
            // LANES x {1, 2, 3}
            while col < n_cols { 
                let col_slice = &a[col * lda + row_base .. col * lda + row_base + mr]; 
                let (achunks, atail) = col_slice.as_chunks::<LANES>();

                let mut acc = Simd::<f32, LANES>::splat(0.0); 
                for (xchunk, achunk) in xchunks.iter().zip(achunks.iter()) { 
                    let xv = Simd::<f32, LANES>::from_array(*xchunk); 
                    let av = Simd::<f32, LANES>::from_array(*achunk); 

                    acc = av.mul_add(xv, acc); 
                }

                let mut sum = acc.reduce_sum();

                for (xt, at) in xtail.iter().zip(atail.iter()) { 
                    sum += *at * *xt; 
                }

                y[col] += sum; 
                col += 1; 
            }   
        } else {
            // short panel 
            for (col, yv) in y.iter_mut().enumerate() { 
                let mut sum = 0.0; 

                let col_beg = row_base + col * lda; 
                let col_end = col_beg + mr; 
                let a_slice = &a[col_beg .. col_end]; 

                for (r, xv) in x_panel.iter().enumerate() { 
                    sum += a_slice[r] * xv; 
                }

                *yv += sum; 
            }
        }
        row_base += mr; 
    }
}

/// Performs a transpose matrix-vector multiply with 
/// no scaling constants. 
///
/// Arguments: 
/// * `a`: [MatrixRef] - over [f32] 
/// * `x`: [VectorRef] - over [f32] 
/// * `y`: [VectorMut] - over [f32] 
///
/// Returns: 
/// Nothing. `y.data` is overwritten. 
#[inline] 
pub fn sdotf ( 
    a: MatrixRef<'_, f32>, 
    x: VectorRef<'_, f32>, 
    mut y: VectorMut<'_, f32> 
) { 

    let n_rows = a.n_rows(); 
    let n_cols = a.n_cols(); 
    let lda = a.lda(); 

    if n_rows == 0 || n_cols == 0 { 
        return; 
    }

    // fast path 
    if let (Some(xs), Some(ys)) = (x.contiguous_slice(), y.contiguous_slice_mut()) { 
        sdotf_contiguous(n_rows, n_cols, xs, a.as_slice(), lda, ys); 
        return; 
    }

    // slow path 
    let incx = x.stride(); 
    let incy = y.stride(); 
    let aoff = a.offset(); 
    let xs = x.as_slice(); 
    let ys = y.as_slice_mut(); 

    for col_idx in 0..n_cols { 
        let col_beg = aoff + col_idx * lda; 
        let col_end = col_beg + n_rows; 
        let col = &a.as_slice()[col_beg .. col_end]; 

        let avec = VectorRef::new(col, n_rows, 1, 0)
            .expect("a view failed"); 
        let xvec = VectorRef::new(xs, n_rows, incx, 0)
            .expect("x view failed"); 

        let sum = sdot(avec, xvec);

        ys[col_idx * incy] += sum; 
    }
}
