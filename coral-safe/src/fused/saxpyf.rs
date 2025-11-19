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

            // values in y to be updated
            // re-used across all columns
            let y_panel = &mut ys[row_idx .. row_idx + mr];

            // block over cols 
            for col_idx in (0..n_cols).step_by(NR) { 
                let nr = (n_cols - col_idx).min(NR); 
                
                // values in x to be used as alpha scale
                // indexed across nr columns 
                let xs_panel = &xs[col_idx .. col_idx + nr]; 
                
                // safe "pointer" to first element in 
                // mr x nr block
                let a_panel = a.panel(row_idx, col_idx, mr, nr);

                // each column in mr x nr block
                // avoids bounds checks with x
                for (j, &alpha) in xs_panel.iter().enumerate() { 
                    if alpha == 0.0 { 
                        continue; 
                    }

                    // gets the jth column in the mr x nr view 
                    // as an immutable slice
                    let acolumn = a_panel.get_column_slice(j); 

                    // `expect` is used to balance BLAS 
                    // with rust idiomacy. 
                    //
                    // GEMV should not return a Result. 
                    let avec = VectorRef::new(acolumn, mr, 1, 0)
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
