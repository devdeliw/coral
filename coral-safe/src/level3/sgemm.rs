//! Level 3 [`?GEMM`](https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm.html) 
//! routine in single precision
//!
//! \\[
//! C \leftarrow \alpha \operatorname{op}(A)\operatorname{op}(B) + \beta C, \quad
//! \operatorname{op}(A) \in \{A, A^{T}\}.
//! \\]
//!
//! A, B, and C are stored in column-major order.
//!
//! # Author 
//! Deval Deliwala


use crate::types::{CoralTranspose, MatrixRef, MatrixMut};

use crate::level3::{
    sgemm_nn::sgemm_nn,
    sgemm_nt::sgemm_nt,
    sgemm_tn::sgemm_tn,
    sgemm_tt::sgemm_tt,
};

pub(crate) const MC: usize = 256;
pub(crate) const NC: usize = 576;
pub(crate) const KC: usize = 512;


/// General matrix-matrix multiply.
/// `C := alpha AB + beta C`
///
/// # Arguments
/// * `op_a` : [CoralTranspose] - whether to transpose `A`.
/// * `op_b` : [CoralTranspose] - whether to transpose `B`.
/// * `alpha`: [f32] - scalar for `op(A) * op(B)`.
/// * `beta` : [f32] - scalar for `C`.
/// * `a`    : [MatrixRef] - over [f32] 
/// * `b`    : [MatrixRef] - over [f32]
/// * `c`    : [MatrixMut] - over [f32] 
///
/// Returns: 
/// Nothing. `c.data` is updated in place. 
#[inline(always)]
pub fn sgemm(
    op_a: CoralTranspose,
    op_b: CoralTranspose,
    alpha: f32,
    beta: f32,
    a: MatrixRef<'_, f32>,
    b: MatrixRef<'_, f32>,
    mut c: MatrixMut<'_, f32>,
) {
    let a_t = op_a.is_transpose();
    let b_t = op_b.is_transpose();

    let (a_rows, a_cols) = (a.n_rows(), a.n_cols());
    let (b_rows, b_cols) = (b.n_rows(), b.n_cols());
    let (c_rows, c_cols) = (c.n_rows(), c.n_cols());

    let (m_from_a, k_from_a) = if !a_t {
        (a_rows, a_cols) 
    } else {
        (a_cols, a_rows) 
    };

    let (k_from_b, n_from_b) = if !b_t {
        (b_rows, b_cols) 
    } else {
        (b_cols, b_rows)
    };

    debug_assert_eq!(
        m_from_a, c_rows,
        "sgemm: mismatch in m; rows of op(A) vs rows of C."
    );
    debug_assert_eq!(
        n_from_b, c_cols,
        "sgemm: mismatch in n; cols of op(B) vs cols of C."
    );
    debug_assert_eq!(
        k_from_a, k_from_b,
        "sgemm: mismatch in inner dimension k; op(A) vs op(B)"
    );

    let m = m_from_a;
    let n = n_from_b;
    let k = k_from_a;

    let lda = a.lda();
    let ldb = b.lda();
    let ldc = c.lda();

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let c_data = c.as_slice_mut();

    match (a_t, b_t) {
        (false, false) => {
            sgemm_nn(m, n, k, alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
        }
        (false, true) => {
            sgemm_nt(m, n, k, alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
        }
        (true, false) => {
            sgemm_tn(m, n, k, alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
        }
        (true, true) => {
            sgemm_tt(m, n, k, alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
        }
    }
}

