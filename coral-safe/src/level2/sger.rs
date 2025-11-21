use std::simd::{Simd, StdFloat}; 
use crate::types::{MatrixMut, VectorRef, VectorMut}; 
use crate::level1::saxpy; 

const MR: usize = 128; 
const NC: usize = 128; 
const NR: usize = 4; 
const LANES: usize = 32; 


#[inline] 
fn sger_contiguous ( 
    n_rows: usize, 
    n_cols: usize, 
    alpha: f32, 
    x: &[f32], 
    a: &mut [f32], 
    lda: usize, 
    y: &[f32], 
) { 
    if alpha == 0.0 { 
        return; 
    }

    debug_assert!(a.len() >= lda * (n_cols - 1) + n_rows );

    let mut col = 0; 
    while col < n_cols { 
        let nb = (n_cols - col).min(NC); 

        let mut row = 0; 
        while row < n_rows { 
            let mr = (n_rows - row).min(MR); 

            // reused across NR cols at a time 
            let x_panel = &x[row .. row + mr]; 
            let (xchunks, xtail) = x_panel.as_chunks::<LANES>(); 
            let n_chunks = xchunks.len(); 

            if n_chunks > 0 { 
                // NR cols at a time
                let mut j = 0; 
                while j + NR <= nb { 
                    let col0 = col + j; 
                    let col1 = col0 + 1; 
                    let col2 = col1 + 1; 
                    let col3 = col2 + 1; 

                    let y0 = y[col0]; 
                    let y1 = y[col1]; 
                    let y2 = y[col2]; 
                    let y3 = y[col3]; 

                    if y0 != 0.0 || y1 != 0.0 || y2 != 0.0 ||  y3 != 0.0 { 
                        let alpha0 = alpha * y0; 
                        let alpha1 = alpha * y1; 
                        let alpha2 = alpha * y2; 
                        let alpha3 = alpha * y3; 

                        // full logical matrix  
                        let start = col0 * lda; 
                        let (_, right) = a.split_at_mut(start); 
                        let (cols_block, _) = right.split_at_mut(lda * NR);

                        // 4 cols from A
                        let (col0_full, rem) = cols_block.split_at_mut(lda); 
                        let (col1_full, rem) = rem.split_at_mut(lda); 
                        let (col2_full, rem) = rem.split_at_mut(lda); 
                        let (col3_full, _)   = rem.split_at_mut(lda);

                        // mr panels 
                        let col0_panel = &mut col0_full[row .. row + mr]; 
                        let col1_panel = &mut col1_full[row .. row + mr]; 
                        let col2_panel = &mut col2_full[row .. row + mr]; 
                        let col3_panel = &mut col3_full[row .. row + mr]; 

                        // four LANES x 1 cols 
                        let (col0_chunks, _) = col0_panel.as_chunks_mut::<LANES>();
                        let (col1_chunks, _) = col1_panel.as_chunks_mut::<LANES>();
                        let (col2_chunks, _) = col2_panel.as_chunks_mut::<LANES>();
                        let (col3_chunks, _) = col3_panel.as_chunks_mut::<LANES>();

                        let alpha0v = Simd::<f32, LANES>::splat(alpha0);
                        let alpha1v = Simd::<f32, LANES>::splat(alpha1);
                        let alpha2v = Simd::<f32, LANES>::splat(alpha2);
                        let alpha3v = Simd::<f32, LANES>::splat(alpha3);

                        for chunk_idx in 0..n_chunks { 
                            // re-used across 4 cols 
                            let xv = Simd::<f32, LANES>::from_array(xchunks[chunk_idx]);

                            if y0 != 0.0 {
                                let mut a0 = Simd::<f32, LANES>::from_array(col0_chunks[chunk_idx]);
                                a0 = xv.mul_add(alpha0v, a0);
                                col0_chunks[chunk_idx] = a0.to_array();
                            }

                            if y1 != 0.0 {
                                let mut a1 = Simd::<f32, LANES>::from_array(col1_chunks[chunk_idx]);
                                a1 = xv.mul_add(alpha1v, a1);
                                col1_chunks[chunk_idx] = a1.to_array();
                            }

                            if y2 != 0.0 {
                                let mut a2 = Simd::<f32, LANES>::from_array(col2_chunks[chunk_idx]);
                                a2 = xv.mul_add(alpha2v, a2);
                                col2_chunks[chunk_idx] = a2.to_array();
                            }

                            if y3 != 0.0 {
                                let mut a3 = Simd::<f32, LANES>::from_array(col3_chunks[chunk_idx]);
                                a3 = xv.mul_add(alpha3v, a3);
                                col3_chunks[chunk_idx] = a3.to_array();
                            }
                        }

                        // leftover rows 0..LANES
                        if !xtail.is_empty() { 
                            let tail_beg = n_chunks * LANES; 

                            for (r, &xr) in xtail.iter().enumerate() {
                                let idx = tail_beg + r;
                                if y0 != 0.0 { col0_panel[idx] += alpha0 * xr; }
                                if y1 != 0.0 { col1_panel[idx] += alpha1 * xr; }
                                if y2 != 0.0 { col2_panel[idx] += alpha2 * xr; }
                                if y3 != 0.0 { col3_panel[idx] += alpha3 * xr; }
                            }
                        }
                    }

                    j += NR; 
                }

                // leftover cols 0..4
                while j < nb {
                    let colj = col + j;
                    let yj   = y[colj];
                    if yj != 0.0 {
                        let alpha_j = alpha * yj;

                        let start = colj * lda;
                        let (_left, right) = a.split_at_mut(start);
                        let (col_full, _rest) = right.split_at_mut(lda);

                        let col_panel = &mut col_full[row .. row + mr];
                        let (col_chunks, _) = col_panel.as_chunks_mut::<LANES>();

                        let alphav = Simd::<f32, LANES>::splat(alpha_j);

                        for (chunk_idx, col_chunk) in col_chunks.iter_mut().enumerate() {
                            let xv = Simd::<f32, LANES>::from_array(xchunks[chunk_idx]);
                            let mut av = Simd::<f32, LANES>::from_array(*col_chunk);
                            av = xv.mul_add(alphav, av);
                            *col_chunk = av.to_array();
                        }

                        if !xtail.is_empty() {
                            let tail_beg = n_chunks * LANES;
                            for (r, &xr) in xtail.iter().enumerate() {
                                let idx = tail_beg + r;
                                col_panel[idx] += alpha_j * xr;
                            }
                        }
                    }

                    j += 1;
                }
            }

            // tiny row panels
            if n_chunks == 0 {
                for j in 0..nb {
                    let colj = col + j;
                    let yj   = y[colj];
                    if yj == 0.0 {
                        continue;
                    }
                    let alpha_j = alpha * yj;

                    let start = colj * lda + row;
                    let col_panel = &mut a[start .. start + mr];

                    for r in 0..mr {
                        col_panel[r] += alpha_j * x_panel[r];
                    }
                }
            }

            row += mr;
        }

        col += nb;
    }
}


#[inline]
pub fn sger(
    alpha: f32,
    mut a: MatrixMut<'_, f32>,
    x: VectorRef<'_, f32>,
    y: VectorRef<'_, f32>,
) {
    let n_rows = a.n_rows();
    let n_cols = a.n_cols();

    debug_assert_eq!(y.n(), n_cols, "logical length of y must equal n_cols");
    debug_assert_eq!(x.n(), n_rows, "logical length of x must equal n_rows");

    if n_rows == 0 || n_cols == 0 {
        return;
    }
    if alpha == 0.0 {
        return;
    }

    let lda   = a.lda();
    let aoff  = a.offset();
    let adata = &mut a.as_slice_mut()[aoff..];

    // fast path
    if let (Some(xs), Some(ys)) = (x.contiguous_slice(), y.contiguous_slice()) {
        sger_contiguous(n_rows, n_cols, alpha, xs, adata, lda, ys);
        return;
    }

    // slow path 
    let incx = x.stride();
    let incy = y.stride();
    let xs = x.as_slice();
    let ys = y.as_slice();

    for j in 0..n_cols {
        let yj = ys[j * incy];
        if yj == 0.0 {
            continue;
        }

        let alpha_j = alpha * yj;

        let col_start = aoff + j * lda;
        let col = &mut adata[col_start .. col_start + n_rows];

        // A[:, j] += alpha_j * x
        let xview = VectorRef::new(xs, n_rows, incx, 0)
            .expect("x view failed");
        let acol = VectorMut::new(col, n_rows, 1, 0)
            .expect("A column view failed");

        saxpy(alpha_j, xview, acol);
    }
}

