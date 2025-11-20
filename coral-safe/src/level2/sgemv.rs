//! Level 2 [`?GEMV`](https://www.netlib.org/lapack/explore-html/d7/dda/group__gemv.html)
//! routine in single precision. 
//!
//! \\[ 
//! y \leftarrow \alpha A x + \beta y 
//! \\] 
//!
//! # Author 
//! Deval Deliwala


use crate::level2::{sgemv_n, sgemv_t}; 
use crate::types::{MatrixRef, VectorRef, VectorMut, CoralTranspose}; 


/// Performs a general matrix-vector multiply in single precision. 
///
/// Arguments: 
/// * `op`: [CoralTranspose] - `A` transpose or not
/// * `alpha`: [f32] - scalar for `alpha * A x` 
/// * `beta` : [f32] - scalar for `beta * y`
/// * `a` : [MatrixRef] - over [f32] 
/// * `x` : [VectorRef] - over [f32] 
/// * `y` : [VectorMut] - over [f32] 
///
/// Returns: 
/// Nothing. `y.data` is overwritten. 
#[inline] 
pub fn sgemv ( 
    op: CoralTranspose, 
    alpha: f32, 
    beta: f32, 
    a: MatrixRef<'_, f32>, 
    x: VectorRef<'_, f32>, 
    y: VectorMut<'_, f32>, 
) { 
    match op { 
        CoralTranspose::NoTrans => sgemv_n ( alpha, beta, a, x, y ), 
        CoralTranspose::Trans   => sgemv_t ( alpha, beta, a, x, y ), 
    }
}   
