//! `GEMM`. Single precision complex general matrix-multiply.
//!
//! \\[ 
//! C := \alpha \operatorname{op}(A) \operatorname{op}(B) + \beta C, \quad
//!    \operatorname{op}(A) \in \\{A, A^T, A^H\\}. 
//! \\]
//!
//! $A$, $B$, and $C$ are stored in column-major order. Complex scalars and
//! matrix elements are represented as interleaved real-imag pairs (`[re, im]`).
//!
//! # Author
//! Deval Deliwala

use crate::level3::{
    cgemm_nn::cgemm_nn,
    cgemm_nt::cgemm_nt,
    cgemm_tn::cgemm_tn,
    cgemm_tt::cgemm_tt,
    cgemm_nc::cgemm_nc,
    cgemm_tc::cgemm_tc,
    cgemm_cn::cgemm_cn,
    cgemm_ct::cgemm_ct,
    cgemm_cc::cgemm_cc,
};

use crate::enums::CoralTranspose;
use crate::level3::microkernel::c32_mrxnr::Complex32; 

pub(crate) const MC: usize = 384;
pub(crate) const NC: usize = 576;
pub(crate) const KC: usize = 256;

#[inline(always)]
fn c32(x: [f32; 2]) -> Complex32 {
    Complex32 { 
        re: x[0], 
        im: x[1]
    }
}

/// General matrix-matrix multiply 
///
/// # Arguments
/// - `op_a`  (CoralTranspose) : Whether to transpose or conjugate-transpose `A`.
/// - `op_b`  (CoralTranspose) : Whether to transpose or conjugate-transpose `B`.
/// - `m`     (usize)          : Number of rows of `op(A)` and `C`.
/// - `n`     (usize)          : Number of columns of `op(B)` and `C`.
/// - `k`     (usize)          : Shared inner dimension of `op(A)` and `op(B)`.
/// - `alpha` ([f32; 2])       : Complex scalar multiplier for `op(A) * op(B)`.
/// - `a`     (*const f32)     : Pointer to matrix `A`.
/// - `lda`   (usize)          : Leading dimension of `A`.
/// - `b`     (*const f32)     : Pointer to matrix `B`.
/// - `ldb`   (usize)          : Leading dimension of `B`.
/// - `beta`  ([f32; 2])       : Complex scalar applied to `C`.
/// - `c`     (*mut f32)       : Pointer to matrix `C`.
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
/// use coral_aarch64::level3::cgemm;
/// use coral_aarch64::enums::CoralTranspose;
///
/// fn main() {
///     // A = [[1 + 2i, 3 + 4i], 
///     //      [5 + 6i, 7 + 8i]]
///     let a = vec![
///         1.0, 2.0, 5.0, 6.0,   // column 0
///         3.0, 4.0, 7.0, 8.0,   // column 1
///     ];
///
///     // B = [[9 + i,  2 + 3i],
///     //      [4 + 5i, 6 + 7i]]
///     let b = vec![
///         9.0, 1.0, 4.0, 5.0,   // column 0
///         2.0, 3.0, 6.0, 7.0,   // column 1
///     ];
///
///     // C = [[1 + i, 0 + 2i], 
///     //      [3 + 0i, 4 + i]]
///     let mut c = vec![
///         1.0, 1.0, 3.0, 0.0,   // column 0
///         0.0, 2.0, 4.0, 1.0,   // column 1
///     ];
///
///     let m = 2;
///     let n = 2;
///     let k = 2;
///
///     let alpha = [2.0, 1.0];  // 2 + i 
///     let beta  = [1.0, -1.0]; // 1 - i
///
///     cgemm(
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
///
///     // C = [[-50 + 99i,  -78 + 92i],
///     //      [-69 + 276i, -163 + 223i]]
///     // C[0,0] = -50 + 99i
///     assert!((c[0] - (-50.0)).abs() < 1e-6);
///     assert!((c[1] - ( 99.0)).abs() < 1e-6);
///
///     // C[1,0] = -69 + 276i
///     assert!((c[2] - (-69.0)).abs() < 1e-6);
///     assert!((c[3] - (276.0)).abs() < 1e-6);
///
///     // C[0,1] = -78 + 92i
///     assert!((c[4] - (-78.0)).abs() < 1e-6);
///     assert!((c[5] - ( 92.0)).abs() < 1e-6);
///
///     // C[1,1] = -163 + 223i
///     assert!((c[6] - (-163.0)).abs() < 1e-6);
///     assert!((c[7] - ( 223.0)).abs() < 1e-6);
/// }
/// ```
#[inline(always)]
pub fn cgemm(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
    m     : usize,
    n     : usize,
    k     : usize,
    alpha : [f32; 2],
    a     : *const f32,
    lda   : usize,
    b     : *const f32,
    ldb   : usize,
    beta  : [f32; 2],
    c     : *mut f32,
    ldc   : usize,
) {
    
    let alpha = c32(alpha); 
    let beta  = c32(beta); 

    match (op_a, op_b) {
        (CoralTranspose::NoTranspose,        CoralTranspose::NoTranspose)        => cgemm_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::NoTranspose,        CoralTranspose::Transpose)          => cgemm_nt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::Transpose,          CoralTranspose::NoTranspose)        => cgemm_tn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::Transpose,          CoralTranspose::Transpose)          => cgemm_tt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::NoTranspose,        CoralTranspose::ConjugateTranspose) => cgemm_nc(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::Transpose,          CoralTranspose::ConjugateTranspose) => cgemm_tc(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::NoTranspose)        => cgemm_cn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::Transpose)          => cgemm_ct(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::ConjugateTranspose) => cgemm_cc(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
    }
}

