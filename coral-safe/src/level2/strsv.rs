//! Level 2 [`?TRSV`](https://www.netlib.org/lapack/explore-html/dd/dc3/group__trsv.html)
//! routine in single precision. 
//!
//! Solves the triangular system 
//!
//! \\[ 
//! \operatorname{op}(A)x = b, \quad \operatorname{op}(A) \in \\{A, A^T \\}. 
//! \\]
//!
//! # Author 
//! Deval Deliwala


use crate::level2::{strusv, strlsv}; 
use crate::types::{CoralDiagonal, CoralTranspose, CoralTriangular, MatrixRef, VectorMut}; 


/// Performs a triangular solve, where `a` is either upper or lower triangular. 
///
/// Arguments: 
/// * `uplo`: [CoralTriangular] - whether `a` upper or lower triangular 
/// * `trans`: [CoralTranspose] - whether `a` is transposed or not 
/// * `diag`: [CoralDiagonal] - whether `a` has a unit-diagonal or not 
/// * `a`: [MatrixRef] - over [f32] 
/// * `x`: [VectorMut] - over [f32], input as `b`, output as solved `x` 
#[inline]
pub fn strsv ( 
    uplo:  CoralTriangular,
    trans: CoralTranspose, 
    diag:  CoralDiagonal, 
    a: MatrixRef<'_, f32>, 
    x: VectorMut<'_, f32>, 
) { 
    match uplo { 
        CoralTriangular::Upper => strusv(trans, diag, a, x), 
        CoralTriangular::Lower => strlsv(trans, diag, a, x),
    }
}
