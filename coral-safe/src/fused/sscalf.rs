//! Level 1.5 
//! [`SSCAL`](https://www.netlib.org/lapack/explore-html/d2/de8/group__scal_ga3b80044a9dbfcbdcb06e48352ee8d64e.html#ga3b80044a9dbfcbdcb06e48352ee8d64e)
//! that scales a matrix by a scalar. 
//!
//! \\[ 
//! A \leftarrow \alpha A 
//! \\]
//!
//! This isn't fused. It just calls [sscal] `n_cols` times. 
//!
//! # Author 
//! Deval Deliwala 

use crate::level1::sscal; 
use crate::types::{MatrixMut, VectorMut};

/// Scales a matrix by alpha. 
///
/// Arguments: 
/// * `alpha`: [f32] - scalar for matrix. 
/// * `a`: [MatrixMut] - over [f32]. 
///
/// Returns: 
/// Nothing. `a` is scaled in place. 
#[inline] 
pub fn sscalf ( 
    alpha: f32, 
    mut a: MatrixMut<'_, f32>, 
) { 
    let m = a.n_rows(); 
    let n = a.n_cols(); 
    let lda = a.lda(); 

    // quick return 
    if alpha == 1.0 || m == 0 || n == 0 { 
        return; 
    } 

    let adata = a.as_slice_mut(); 
    for j in 0..n { 
        let col = &mut adata[j * lda .. j * lda + m];
        let view = VectorMut::new(col, m, 1, 0)
            .expect("VectorMut::new failed");

        sscal(alpha, view); 
    }
}
