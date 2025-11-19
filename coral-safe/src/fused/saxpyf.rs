//! Level 1.5 (fused) `SAXPYF` routine. Basicaly a GEMV.  
//!
//! \\[ 
//! y \leftarrow y + A x
//! \\]
//!
//! # Author 
//! Deval Deliwala


use crate::level1::{saxpy, sdot}; 
use crate::types::{VectorMut, VectorRef, MatrixRef}; 

const MR: usize = 128; 
const NR: usize = 128; 


/// Performs a fused SAXPY operation on a matrix view `a` of the form 
/// y := y + Ax 
///
/// Arguments: 
/// * a: [MatrixRef] - over [f32] 
/// * x: [VectorRef] - over [f32] 
/// * y: [VectorMut] - over [f32] 
///
/// Returns: 
/// Nothing. `y.data` is overwritten. 
#[inline] 
pub fn saxpyf ( 
    a: MatrixRef<'_, f32>, 
    x: VectorRef<'_, f32>, 
    mut y: VectorMut<'_, f32>, 
) { 
    let n_rows = a.n_rows(); 
    let n_cols = a.n_cols();

    debug_assert_eq!(x.n(), n_cols, "x len must equal # cols in A"); 
    debug_assert_eq!(y.n(), n_rows, "y len must equal # rows in A");  

    if n_rows == 0 || n_cols == 0 { 
        return; 
    }

    let a_data = a.as_slice(); 
    let lda    = a.lda(); 
    let a_offs = a.offset();
    
    // fast path
    if let (Some(xs), Some(ys)) = (x.contiguous_slice(), y.contiguous_slice_mut()) {
        // block over rows
        for row_idx in (0..n_rows).step_by(MR) { 
            let mr = (n_rows - row_idx).min(MR); 

            // gets updated 
            let y_panel = &mut ys[row_idx .. row_idx + mr]; 

            // block over cols 
            for col_idx in (0..n_cols).step_by(NR) { 
                let nr = (n_cols - col_idx).min(NR); 

                // each column in col block
                for j in 0..nr { 
                    let col = col_idx + j; 

                    let alpha = xs[col]; 
                    if alpha == 0.0 { 
                        continue; 
                    } 

                    let col_beg = a_offs + row_idx + col * lda; 
                    let col_end = col_beg + mr; 
                    let abuf = &a_data[col_beg .. col_end]; 

                    // `expect` is used to balance BLAS 
                    // with rust idiomacy. 
                    //
                    // GEMV should not return a Result. 
                    let avec = VectorRef::new(abuf, mr, 1, 0)
                        .expect("a column view is out of bounds"); 
                    let yvec = VectorMut::new(y_panel, mr, 1, 0)
                        .expect("y panel view is out of bounds"); 

                    // y_panel = alpha * A y
                    saxpy(alpha, avec, yvec); 
                }
            }
        }

        return;
    }

    // slow path 
    let yoff = y.offset(); 
    let incy = y.stride();
    let ybuf = y.as_slice_mut(); 

    for row in 0..n_rows  { 
        let a_row = VectorRef::new(a_data, n_cols, lda, a_offs + row)
            .expect("a row view is out of bounds");

        let value = sdot(a_row, x); 
        let y_idx = yoff + row * incy; 
        ybuf[y_idx] += value; 
    }
}
