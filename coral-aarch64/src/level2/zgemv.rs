//! `GEMV`. General double precision complex matrix-vector multiply.
//!
//! \\[ 
//! y := \alpha \operatorname{op}(A) x + \beta y, \quad \operatorname{op}(A) \in \\{A, A^{T}, A^{H}\\}.
//! \\]
//!
//! `A` is interleaved column-major `[re, im, ...]`
//!
//! # Arguments
//! - `n_rows` (usize)      : Number of rows ($m$) in the matrix $A$.
//! - `n_cols` (usize)      : Number of columns ($n$) in the matrix $A$.
//! - `alpha`  ([f64; 2])   : Complex scalar multiplier applied to the product $A^T x$.
//! - `matrix` (&[f64])     : Input slice containing the matrix $A$.
//! - `lda`    (usize)      : Leading dimension of $A$.
//! - `x`      (&[f64])     : Input complex vector of length `n_rows`.
//! - `incx`   (usize)      : Stride between consecutive complex elements of $x$.
//! - `beta`   ([f64; 2])   : Complex scalar multiplier applied to $y$ prior to accumulation.
//! - `y`      (&mut [f64]) : Input/output complex vector of length `n_cols`.
//! - `incy`   (usize)      : Stride between consecutive complex elements of $y$.
//!
//! # Returns
//! - Nothing. The contents of $y$ are updated in place.
//! 
//! # Example
//! ```rust
//! use coral_aarch64::level2::zgemv;
//! use coral_aarch64::enums::CoralTranspose;
//!
//! fn main() {
//!     let m = 2;
//!     let n = 2;
//!     let op = CoralTranspose::NoTranspose;
//!
//!     let a = vec![
//!         1.0, 0.0, 0.0, 1.0, // col 0 
//!         2.0, 0.0, 0.0, 0.0, // col 1
//!     ];
//!
//!     let lda   = m;
//!
//!     let x     = vec![1.0, -1.0,  0.5, 0.5]; // (1 - i, 0.5 + 0.5i)
//!     let incx  = 1;
//!     let mut y = vec![0.0, 0.0,  0.0, 0.0];
//!     let incy  = 1;
//!
//!     let alpha = [1.0, 0.0];
//!     let beta  = [0.0, 0.0];
//!
//!     zgemv(op, m, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
//! }
//! ```


use crate::enums::CoralTranspose; 
use crate::level2::{ 
    zgemv_notranspose::zgemv_notranspose, 
    zgemv_transpose::zgemv_transpose, 
    zgemv_conjtranspose::zgemv_conjtranspose
}; 

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn zgemv( 
    trans   : CoralTranspose, 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : [f64; 2], 
    matrix  : &[f64], 
    lda     : usize, 
    x       : &[f64], 
    incx    : usize, 
    beta    : [f64; 2], 
    y       : &mut [f64], 
    incy    : usize
) { 
    match trans { 
        CoralTranspose::NoTranspose         => zgemv_notranspose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::Transpose           => zgemv_transpose    (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::ConjugateTranspose  => zgemv_conjtranspose(n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
    }
}
