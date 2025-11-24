//! `GEMV`. General single precision matrix-vector multiply.
//!
//! \\[
//! y := \alpha \operatorname{op}(A) x + \beta y, \quad \operatorname{op}(A) \in \\{A, A^{T}\\}.
//! \\]
//!
//! # Author
//! Deval Deliwala

use crate::enums::CoralTranspose; 
use crate::level2::{ 
    sgemv_transpose::sgemv_transpose, 
    sgemv_notranspose::sgemv_notranspose, 
}; 

/// General matrix-vector multiply 
///
/// # Arguments
/// - `trans`  (CoralTranspose) : Whether $A$ is $A$ or $A^T$. 
/// - `n_rows` (usize)      : Number of rows ($m$) in the matrix $A$.
/// - `n_cols` (usize)      : Number of columns ($n$) in the matrix $A$.
/// - `alpha`  (f32)        : Scalar multiplier applied to the product $A x$.
/// - `matrix` (&[f32])     : Input slice containing the matrix $A$.
/// - `lda`    (usize)      : Leading dimension of $A$.
/// - `x`      (&[f32])     : Input vector of length `n_cols`.
/// - `incx`   (usize)      : Stride between consecutive elements of $x$.
/// - `beta`   (f32)        : Scalar multiplier applied to $y$ prior to accumulation.
/// - `y`      (&mut [f32]) : Input/output vector of length `n_rows`.
/// - `incy`   (usize)      : Stride between consecutive elements of $y$.
///
/// # Returns
/// - Nothing. The contents of $y$ are updated in place.
/// 
/// # Example
/// ```rust
/// use coral_aarch64::level2::sgemv;
/// use coral_aarch64::enums::CoralTranspose; 
///
/// fn main() {
///     let m  = 2;
///     let n  = 3;
///     let op = CoralTranspose::NoTranspose;
///
///     let a = vec![
///         1.0, 2.0,  // col 0
///         3.0, 4.0,  // col 1
///         5.0, 6.0,  // col 2
///     ];
///
///     let lda   = m;
///     let x     = vec![1.0, 2.0, 3.0];
///     let incx  = 1;
///     let mut y = vec![0.5, -1.0];
///     let incy  = 1;
///
///     let alpha = 2.0;
///     let beta  = 0.5;
///
///     sgemv(op, m, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
/// }
/// ```
#[inline]
#[cfg(target_arch = "aarch64")]
pub fn sgemv( 
    trans   : CoralTranspose, 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : f32, 
    matrix  : &[f32], 
    lda     : usize, 
    x       : &[f32], 
    incx    : usize, 
    beta    : f32, 
    y       : &mut [f32], 
    incy    : usize
) { 
    match trans { 
        CoralTranspose::NoTranspose         => sgemv_notranspose(n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::Transpose           => sgemv_transpose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::ConjugateTranspose  => sgemv_transpose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
    }
}
