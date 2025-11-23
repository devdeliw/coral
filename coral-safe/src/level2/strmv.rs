//! Level 2 [`?TRMV`](https://www.netlib.org/lapack/explore-html/d6/d1c/group__trmv.html)
//! routine in single precision. 
//!
//! //[ 
//! x \leftarrow \operatorname{op}(A)x, \quad \operatorname{op}{A} \in \\{A, A^T \\}. 
//! //]
//!
//! # Author 
//! Deval Deliwala


use crate::level2::{strlmv, strumv}; 
use crate::types::{CoralDiagonal, CoralTranspose, CoralTriangular, MatrixRef, VectorMut}; 


/// Performs a triangular matrix-vector multiply. 
///
/// Arguments: 
/// * `uplo`: [CoralTriangular] - whether `a` upper or lower triangular  
/// * `trans`: [CoralTranspose] - whether `a` is transposed or not 
/// * `diag`: [CoralDiagonal] - whether `a` has a unit-diagonal or not 
/// * `a`: [MatrixRef] - over [f32] 
/// * `x`: [VectorMut] - over [f32], input as 'x', output as result `b`
///
/// Returns: 
/// Nothing. `x.data` updated in place. 
#[inline] 
pub fn strmv ( 
    uplo: CoralTriangular, 
    trans: CoralTranspose, 
    diag: CoralDiagonal, 
    a: MatrixRef<'_, f32>, 
    x: VectorMut<'_, f32>, 
) { 
    match uplo { 
        CoralTriangular::Upper => strumv(trans, diag, a, x), 
        CoralTriangular::Lower => strlmv(trans, diag, a, x), 
    }
}   
