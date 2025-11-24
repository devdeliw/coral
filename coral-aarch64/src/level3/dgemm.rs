//! `GEMM`. Double precision general matrix-multiply.
//!
//! \\[ 
//! C := \alpha \operatorname{op}(A) \operatorname{op}(B) + \beta  C, \quad
//! \operatorname{op}(A) \in \\{A, A^T\\}.
//! \\]
//!
//! $A$, $B$, and $C$ are stored in column-major order.
//!
//! # Author 
//! Deval Deliwala

use crate::level3::{
    dgemm_nn::dgemm_nn, 
    dgemm_nt::dgemm_nt, 
    dgemm_tn::dgemm_tn, 
    dgemm_tt::dgemm_tt
};
use crate::enums::CoralTranspose;

pub(crate) const MC: usize = 258; 
pub(crate) const NC: usize = 384; 
pub(crate) const KC: usize = 256;

#[inline(always)]
fn is_transpose(op: CoralTranspose) -> bool {
    match op {
        CoralTranspose::NoTranspose        => false,
        CoralTranspose::Transpose          => true,
        CoralTranspose::ConjugateTranspose => true, 
    }
}


/// General matrix-matrix multiply 
///
/// # Arguments
/// - `op_a`  (CoralTranspose) : Whether to transpose `A`.
/// - `op_b`  (CoralTranspose) : Whether to transpose `B`.
/// - `m`     (usize)          : Number of rows of `op(A)` and `C`.
/// - `n`     (usize)          : Number of columns of `op(B)` and `C`.
/// - `k`     (usize)          : Shared inner dimension of `op(A)` and `op(B)`.
/// - `alpha` (f64)            : Scalar multiplier for `op(A) * op(B)`.
/// - `a`     (*const f64)     : Pointer to matrix `A`.
/// - `lda`   (usize)          : Leading dimension of `A`.
/// - `b`     (*const f64)     : Pointer to matrix `B`.
/// - `ldb`   (usize)          : Leading dimension of `B`.
/// - `beta`  (f64)            : Scalar multiplier for `C`.
/// - `c`     (*mut f64)       : Pointer to matrix `C`.
/// - `ldc`   (usize)          : Leading dimension of `C`.
///
/// # Returns
/// - Nothing. The contents of `C` are updated in place.
///
/// # Author 
/// Deval Deliwala
///
/// # Example
/// ```rust
/// use coral_aarch64::level3::dgemm;
/// use coral_aarch64::enums::CoralTranspose;
///
/// fn main() {
///     // A = [[1, 3],
///     //      [2, 4]]
///     let a = vec![
///         1.0, 2.0,   // column 0
///         3.0, 4.0,   // column 1
///     ];
///
///     // B = [[5, 7],
///     //      [6, 8]]
///     let b = vec![
///         5.0, 6.0,   // column 0
///         7.0, 8.0,   // column 1
///     ];
///
///     // C = Identity
///     let mut c = vec![
///         1.0, 0.0,
///         0.0, 1.0,
///     ];
///
///     let m = 2;
///     let n = 2;
///     let k = 2;
///
///     let alpha = 2.0;
///     let beta  = 1.0;
///
///     dgemm(
///         CoralTranspose::NoTranspose,
///         CoralTranspose::NoTranspose,
///         m, n, k,
///         alpha,
///         a.as_ptr(), m,
///         b.as_ptr(), k,
///         beta,
///         c.as_mut_ptr(), m,
///     );
///
///     // C = [[47, 62],
///     //      [68, 93]]
///     assert!((c[0] - 47.0).abs() < 1e-12);
///     assert!((c[1] - 68.0).abs() < 1e-12);
///     assert!((c[2] - 62.0).abs() < 1e-12);
///     assert!((c[3] - 93.0).abs() < 1e-12);
/// }
/// ```
#[inline(always)]
pub fn dgemm(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
    m     : usize,
    n     : usize,
    k     : usize,
    alpha : f64,
    a     : *const f64,
    lda   : usize,
    b     : *const f64,
    ldb   : usize,
    beta  : f64,
    c     : *mut f64,
    ldc   : usize,
) {
    let a_t = is_transpose(op_a);
    let b_t = is_transpose(op_b);

    match (a_t, b_t) {
        (false, false) => dgemm_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (false, true ) => dgemm_nt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (true , false) => dgemm_tn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (true , true ) => dgemm_tt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
    }
}

