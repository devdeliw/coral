use std::simd::{Simd, StdFloat};

use crate::types::{MatrixRef, VectorRef, VectorMut}; 
use crate::level1::saxpy::saxpy;

const MR:    usize = 256; 
const LANES: usize = 32; 

#[inline(always)]
fn saxpyf_contiguous(
    n_rows: usize,
    n_cols: usize,
    x: &[f32],
    a: &[f32],
    lda: usize,
    y: &mut [f32],
) {
    // number of columns fused
    const COL_BLOCK: usize = 4;

    // MR x n_cols
    let mut row_base = 0;
    while row_base < n_rows {
        let mr = (n_rows - row_base).min(MR); 

        // y panel view to be updated
        let y_panel = &mut y[row_base .. row_base + mr];

        let (ychunks, ytail) = y_panel.as_chunks_mut::<LANES>(); 
        let n_chunks = ychunks.len(); 

        if n_chunks > 0 { 
            // LANES x COL_BLOCK
            let mut col = 0; 
            while col + COL_BLOCK <= n_cols { 
                let x0 = x[col]; 
                let x1 = x[col + 1]; 
                let x2 = x[col + 2]; 
                let x3 = x[col + 3]; 

                let x0v = Simd::<f32, LANES>::splat(x0); 
                let x1v = Simd::<f32, LANES>::splat(x1); 
                let x2v = Simd::<f32, LANES>::splat(x2); 
                let x3v = Simd::<f32, LANES>::splat(x3); 
                if x0 != 0.0 || x1 != 0.0 || x2 != 0.0 || x3 != 0.0 { 
                    let col0 = &a[(col) * lda + row_base .. (col) * lda + row_base + mr];
                    let col1 = &a[(col + 1) * lda + row_base .. (col + 1) * lda + row_base + mr];
                    let col2 = &a[(col + 2) * lda + row_base .. (col + 2) * lda + row_base + mr];
                    let col3 = &a[(col + 3) * lda + row_base .. (col + 3) * lda + row_base + mr];

                    let (col0_chunks, _) = col0.as_chunks::<LANES>(); 
                    let (col1_chunks, _) = col1.as_chunks::<LANES>(); 
                    let (col2_chunks, _) = col2.as_chunks::<LANES>(); 
                    let (col3_chunks, _) = col3.as_chunks::<LANES>();

                    // fused FMAs across COL_BLOCK rows at a time 
                    // overwrites first n_rows - n_rows % MR elements of y_panel 
                    for (chunk_idx, ychunk) in ychunks.iter_mut().enumerate() { 
                        let mut yv = Simd::<f32, LANES>::from_array(*ychunk);
                        
                        if x0 != 0.0 { 
                            let a0 = Simd::<f32, LANES>::from_array(col0_chunks[chunk_idx]);
                            yv = a0.mul_add(x0v, yv); 
                        }  

                        if x1 != 0.0 { 
                            let a1 = Simd::<f32, LANES>::from_array(col1_chunks[chunk_idx]); 
                            yv = a1.mul_add(x1v, yv); 
                        }

                        if x2 != 0.0 { 
                            let a2 = Simd::<f32, LANES>::from_array(col2_chunks[chunk_idx]); 
                            yv = a2.mul_add(x2v, yv); 
                        }

                        if x3 != 0.0 { 
                            let a3 = Simd::<f32, LANES>::from_array(col3_chunks[chunk_idx]); 
                            yv = a3.mul_add(x3v, yv);
                        }

                        *ychunk = yv.to_array(); 
                    }
                }

                col += COL_BLOCK; 
            }

            // 3 leftover columns 
            while col < n_cols { 
                let alpha = x[col]; 
                if alpha == 0.0 { 
                    continue; 
                }

                let col_beg = row_base + col * lda; 
                let col_end = col_beg + mr; 
                let col_slice = &a[col_beg .. col_end]; 
                let (col_chunks, _) = col_slice.as_chunks::<LANES>();

                let alpha = Simd::<f32, LANES>::splat(alpha); 
                for (chunk_idx, ychunk) in ychunks.iter_mut().enumerate() { 
                    let mut yv = Simd::<f32, LANES>::from_array(*ychunk); 
                    let achunk = Simd::<f32, LANES>::from_array(col_chunks[chunk_idx]); 

                    yv = achunk.mul_add(alpha, yv); 
                    *ychunk = yv.to_array(); 
                }

                col += 1; 
            }
        }

        // leftover rows 
        // n_rows - n_rows % MR 
        if !ytail.is_empty() { 
            let tail_beg = row_base + n_chunks * LANES; 
            let tail_len = ytail.len(); 

            for r in 0..tail_len { 
                let row_idx = tail_beg + r; 
                let mut acc = y[row_idx]; 

                // sweep over all cols 
                for (col_idx, &alpha) in x.iter().enumerate() { 
                    if alpha == 0.0 { 
                        continue; 
                    }

                    acc += alpha * a[row_idx + col_idx * lda]; 
                }

                y[row_idx] = acc; 
            }
        }

        row_base += mr; 
    }
}

#[inline]
pub fn saxpyf(
    a: MatrixRef<'_, f32>, 
    x: VectorRef<'_, f32>, 
    mut y: VectorMut<'_, f32>, 
) {

    let n_rows = a.n_rows(); 
    let n_cols = a.n_cols(); 
    let lda = a.lda(); 

    if n_rows == 0 || n_cols == 0 { 
        return;
    }

    // fast path 
    if let (Some(xs), Some(ys)) = (x.contiguous_slice(), y.contiguous_slice_mut()) { 
        saxpyf_contiguous(n_rows, n_cols, xs, a.as_slice(), lda, ys); 
        return; 
    }

    // slow path 
    let incx = x.stride(); 
    let incy = y.stride(); 

    for col_idx in 0..n_cols {
        let alpha = x.as_slice()[col_idx * incx];
        if alpha != 0.0 {
            let col_start = col_idx * lda;
            let col = &a.as_slice()[col_start .. col_start + n_rows];

            let avec = VectorRef::new(col, n_rows, 1, 0)
                .expect("a view faied"); 
            let yvec = VectorMut::new(y.as_slice_mut(), n_rows, incy, 0)
                .expect("y view failed"); 

            saxpy(alpha, avec, yvec);
        }
    }
}

