//! `TRMV`. Performs a single precision triangular matrix–vector multiply.
//!
//! \\[ 
//! x := \operatorname{op}(A) x, \quad \operatorname{op}(A) \in \\{A, A^{T}\\}. 
//! \\]
//!
//! This function implements the BLAS [`strmv`] routine for both upper and lower
//! triangular matrices.
//!
//! # Author
//! Deval Deliwala


use crate::enums::{CoralDiagonal, CoralTranspose, CoralTriangular};
use crate::level2::{ 
    strlmv::strlmv, 
    strumv::strumv, 
}; 

/// Triangular matrix-vector multiply 
///
/// # Arguments
/// - `uplo`        (CoralTriangular) : Indicates whether $A$ is upper or lower triangular.
/// - `transpose`   (CoralTranspose)  : Specifies whether to use $A$ or $A^T$.
/// - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
/// - `n`           (usize)           : Order of the square matrix $A$.
/// - `matrix`      (&[f32])          : Input slice containing the triangular matrix $A$. 
/// - `lda`         (usize)           : Leading dimension of $A$.
/// - `x`           (&mut [f32])      : Input/output slice containing the vector $x$.
/// - `incx`        (usize)           : Stride between consecutive elements of $x$.
///
/// # Returns
/// - Nothing. The contents of $x$ are updated in place. 
///
/// # Example
/// ```rust
/// use coral_aarch64::level2::strmv;
/// use coral_aarch64::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
///
/// fn main() {
///     let n = 2;
///     let uplo      = CoralTriangular::UpperTriangular;
///     let transpose = CoralTranspose::NoTranspose;
///     let diagonal  = CoralDiagonal::NonUnitDiagonal;
///
///     let a = vec![
///         2.0, 0.0,   // col 0
///         1.0, 3.0,   // col 1
///     ];
///
///     let lda   = n;
///     let mut x = vec![1.0, 2.0];
///     let incx  = 1;
///
///     strmv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
/// }
/// ```
#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn strmv( 
    uplo        : CoralTriangular, 
    transpose   : CoralTranspose, 
    diagonal    : CoralDiagonal, 
    n           : usize, 
    matrix      : &[f32], 
    lda         : usize, 
    x           : &mut [f32], 
    incx        : usize, 
) { 
    match uplo { 
        CoralTriangular::UpperTriangular => strumv(n, diagonal, transpose, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => strlmv(n, diagonal, transpose, matrix, lda, x, incx), 
    }
}
