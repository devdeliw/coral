use crate::fused::saxpyf; 
use crate::level2::pack_vector;
use crate::types::{MatrixRef, VectorRef, VectorMut}; 
use crate::level2::{
    pack_panel::pack_panel, 
    pack_vector::pack_vector_f32
}; 

const MC: usize= 128; 
const NC: usize = 128; 


#[inline] 
pub(crate) fn sgemv_n ( 
    alpha: f32, 
    beta: f32, 
    a: MatrixRef<'_, f32>, 
    x: VectorRef<'_, f32>, 
    mut y: VectorMut<'_, f32>, 
) { 
    let n_cols = a.n_cols(); 
    let n_rows = a.n_rows(); 

    if n_cols == 0 || n_rows == 0 { 
        return; 
    } 

    if alpha == 0.0 && beta == 1.0 { 
        return; 
    } 

    let incx = x.stride(); 
    let incy = y.stride(); 
    let xoff = x.offset(); 
    let yoff = y.offset(); 

    // scale and pack into contiguous buffers
    let mut ybuf = Vec::new(); 
    let mut xbuf = Vec::new(); 
    pack_vector_f32(beta,  n_rows, y.as_slice(), incy, &mut ybuf); 
    pack_vector_f32(alpha, n_cols, x.as_slice(), incx, &mut xbuf); 

    let lda = a.lda(); 
    // fast path 
    if lda == n_rows { 
        let xvec = VectorRef::new(&xbuf, n_cols, 1, 0).expect("x vec view"); 
        let yvec = VectorMut::new(&mut ybuf, n_rows, 1, 0).expect("y vec view"); 
        saxpyf(a, xvec, yvec); 
    } else { 
        let mut apack: Vec<f32> = Vec::new(); 

        let mut row_idx = 0; 
        while row_idx < n_rows { 
            let mb = (n_rows - row_idx).min(MC); 

            let y_sub = &mut ybuf[row_idx .. row_idx + mb];
            let a_sub = &a.as_slice()[row_idx .. (n_cols - 1) * lda + (n_rows - row_idx)];

            let mut col_idx = 0; 
            while col_idx < n_cols { 
                let nb = (n_cols - col_idx).min(NC); 

                pack_panel ( 
                    &mut apack, 
                    a_sub, 
                    mb, 
                    nb, 
                    col_idx, 
                    lda
                ); 

                let aview = MatrixRef::new(&apack, mb, nb, lda, 0)
                    .expect("a mat view"); 
                let xview = VectorRef::new(&xbuf[col_idx .. col_idx + nb], nb, incx, xoff)
                    .expect("x vec view");
                let yview = VectorMut::new(y_sub, mb, incy, yoff)
                    .expect("y vec view"); 
                saxpyf ( 
                    aview, 
                    xview, 
                    yview, 
                );

                col_idx += nb; 
            }

            row_idx += mb; 
        }     
    }

    if incy != 0 { 
        let mut ys = y.as_slice_mut(); 
        let ys_it = ys.iter_mut().step_by(incy).take(n_rows); 

        for (ynew, &yold) in ys_it.zip(ybuf.iter()) { 
            *ynew = yold; 
        }
    }

}


