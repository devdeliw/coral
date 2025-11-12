//! `GEMV`. General single precision complex matrix-vector multiply. 
//!
//! \\[ 
//! y := \alpha \operatorname{op}(A) x + \beta y,
//! \quad \operatorname{op}(A) \in \\{A, A^{T}, A^{H}\\}. 
//! \\]
//!
//! $A$ is an interleaved column-major matrix. `[re, im, ...]`
//!
//! # Arguments
//! - `trans`  (CoralTranspose) : Whether $A$ is $A$, $A^T$ or $A^H$. 
//! - `n_rows` (usize)      : Number of rows ($m$) in the matrix $A$.
//! - `n_cols` (usize)      : Number of columns ($n$) in the matrix $A$.
//! - `alpha`  ([f32; 2])   : Complex scalar multiplier applied to the product $Ax$.
//! - `matrix` (&[f32])     : Input slice containing the interleaved matrix $A$.
//! - `lda`    (usize)      : Leading dimension of $A$; complex units. 
//! - `x`      (&[f32])     : Input complex vector of length `n_cols`.
//! - `incx`   (usize)      : Stride between consecutive complex elements of $x$; complex units. 
//! - `beta`   ([f32; 2])   : Complex scalar multiplier applied to $y$ prior to accumulation.
//! - `y`      (&mut [f32]) : Input/output complex vector of length `n_rows`.
//! - `incy`   (usize)      : Stride between consecutive complex elements of $y$; complex units. 
//!
//! # Returns
//! - Nothing. The contents of $y$ are updated in place.
//!
//! # Author 
//! Deval Deliwala
//! 
//! # Example
//!```rust
//! use coral_aarch64::level2::cgemv;
//! use coral_aarch64::enums::CoralTranspose;
//!
//! fn main() {
//!     let m  = 2;
//!     let n  = 2;
//!     let op = CoralTranspose::NoTranspose;
//!
//!     let a = vec![
//!         1.0, 0.0,  0.0, 1.0,  // col 0: (1+0i, 0+1i)
//!         2.0, 0.0,  0.0, 0.0,  // col 1: (2+0i, 0+0i)
//!     ];
//!
//!     let lda   = m;
//!     let x     = vec![1.0, 1.0, 0.0, -1.0];  // (1+i, -i)
//!     let incx  = 1;
//!     let mut y = vec![0.0, 0.0, 0.0, 0.0];   // 2 outputs
//!     let incy  = 1;
//!
//!     let alpha = [1.0, 0.0];
//!     let beta  = [0.0, 0.0];
//!
//!     cgemv(op, m, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
//! }
//! ```


use crate::enums::CoralTranspose;
use crate::level2::{ 
    cgemv_transpose::cgemv_transpose, 
    cgemv_notranspose::cgemv_notranspose, 
    cgemv_conjtranspose::cgemv_conjtranspose
}; 

#[inline]
#[cfg(target_arch = "aarch64")]
pub fn cgemv( 
    trans   : CoralTranspose, 
    n_rows  : usize, 
    n_cols  : usize, 
    alpha   : [f32; 2], 
    matrix  : &[f32], 
    lda     : usize, 
    x       : &[f32], 
    incx    : usize, 
    beta    : [f32; 2], 
    y       : &mut [f32], 
    incy    : usize
) { 
    match trans { 
        CoralTranspose::NoTranspose         => cgemv_notranspose  (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::Transpose           => cgemv_transpose    (n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
        CoralTranspose::ConjugateTranspose  => cgemv_conjtranspose(n_rows, n_cols, alpha, matrix, lda, x, incx, beta, y, incy),
    }
}
