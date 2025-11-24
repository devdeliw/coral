//! `TRSV`. Performs a single precision triangular solve.
//!
//! \\[ 
//! \operatorname{op}(A) x = b, \quad \operatorname{op}(A) \in \\{A, A^{T}\\}.
//! \\]
//!
//! This function implements the BLAS [`strsv`] routine for both upper and lower
//! triangular systems.
//!
//! # Author 
//! Deval Deliwala 


use crate::enums::{CoralDiagonal, CoralTranspose, CoralTriangular};
use crate::level2::{
    strlsv::strlsv, 
    strusv::strusv,
}; 

/// Triangular solve 
///
/// # Arguments
/// - `uplo`        (CoralTriangular) : Indicates whether $A$ is upper or lower triangular.
/// - `transpose`   (CoralTranspose)  : Specifies whether to solve with $A$ or $A^T$.
/// - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
/// - `n`           (usize)           : Order of the square matrix $A$.
/// - `matrix`      (&[f32])          : Input slice containing the triangular matrix $A$. 
/// - `lda`         (usize)           : Leading dimension of $A$.
/// - `x`           (&mut [f32])      : Input/output slice containing the right-hand side $b$ on entry and the solution $x$ on exit. 
/// - `incx`        (usize)           : Stride between consecutive elements of $x$.
///
/// # Returns
/// - Nothing. $x$ is updated in place with the solution. 
/// 
/// # Example
/// ```rust
/// use coral_aarch64::level2::strsv;
/// use coral_aarch64::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
///
/// fn main() {
///     let n = 2;
///     let uplo = CoralTriangular::UpperTriangular;
///     let transpose = CoralTranspose::NoTranspose;
///     let diagonal  = CoralDiagonal::NonUnitDiagonal;
///
///     let a = vec![
///         2.0, 0.0,  // col 0
///         1.0, 3.0,  // col 1
///     ];
///
///     let lda  = n;
///     let mut x = vec![2.0, 3.0]; // b on entry;
///                                 // overwritten with solution x
///     let incx = 1;
///
///     strsv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
/// }
/// ```
#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn strsv( 
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
        CoralTriangular::UpperTriangular => strusv(n, transpose, diagonal, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => strlsv(n, transpose, diagonal, matrix, lda, x, incx), 
    }
}
