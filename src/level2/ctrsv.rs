//! `TRSV`. Performs a single precision complex triangular solve.
//!
//! \\[ 
//! \operatorname{op}(A) x = b, \quad \operatorname{op}(A) \in \\{A, A^{T}, A^{H}\\}. 
//! \\]
//!
//! This function implements the BLAS [`ctrsv`] routine for both upper and lower
//! triangular systems.
//!
//! # Arguments
//! - `uplo`        (CoralTriangular) : Indicates whether $A$ is upper or lower triangular.
//! - `transpose`   (CoralTranspose)  : Specifies whether to solve with $A$, $A^T$, or $A^H$.
//! - `diagonal`    (CoralDiagonal)   : Indicates if the diagonal is unit (all 1s) or non-unit.
//! - `n`           (usize)           : Order of the square matrix $A$.
//! - `matrix`      (&[f32])          : Input slice containing the interleaved triangular matrix $A$.
//! - `lda`         (usize)           : Leading dimension of $A$.
//! - `x`           (&mut [f32])      : Input/output slice containing the right-hand side $b$ on
//!                                     entry and exits as solution $x$. 
//! - `incx`        (usize)           : Stride between consecutive complex elements of $x$.
//!
//! # Returns
//! - Nothing. $x$ is updated in place with the solution. 
//!
//! # Author
//! Deval Deliwala
//! 
//! # Example
//! ```rust
//! use coral::level2::ctrsv;
//! use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
//!
//! fn main() {
//!     let n = 2;
//!     let uplo      = CoralTriangular::UpperTriangular;
//!     let transpose = CoralTranspose::NoTranspose;
//!     let diagonal  = CoralDiagonal::NonUnitDiagonal;
//!
//!     let a = vec![
//!         2.0, 0.0, 0.0, 0.0,  // col 0: (2+0i, 0+0i)
//!         1.0, 1.0, 3.0, 0.0,  // col 1: (1+i, 3+0i)
//!     ];
//!
//!     let lda   = n;
//!     let mut x = vec![3.0, 0.0,  1.0, 0.0]; // b -> x 
//!     let incx  = 1;
//!
//!     ctrsv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
//! }


use crate::enums::{CoralDiagonal, CoralTranspose, CoralTriangular};
use crate::level2::{
    ctrlsv::ctrlsv, 
    ctrusv::ctrusv,
}; 

#[inline] 
#[cfg(target_arch = "aarch64")]
pub fn ctrsv( 
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
        CoralTriangular::UpperTriangular => ctrusv(n, transpose, diagonal, matrix, lda, x, incx), 
        CoralTriangular::LowerTriangular => ctrlsv(n, transpose, diagonal, matrix, lda, x, incx), 
    }
}
