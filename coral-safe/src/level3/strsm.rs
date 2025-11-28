//! Level 3 [`?TRSM`](https://www.netlib.org/lapack/explore-html/d9/de5/group__trsm.html) 
//! routine in single precision. 
//!
//! \\[ 
//! \operatorname{op}(A) X = \alpha B 
//! \\] 
//! or 
//! \\[ 
//! X \operatorname{op}(A) = \alpha B
//! \\]
//!
//! # Author 
//! Deval Deliwala


use crate::types::{
    MatrixRef, MatrixMut,
    CoralSide,
    CoralTranspose,
    CoralTriangular,
    CoralDiagonal,
};
use crate::level3::{
    sgemm::MC, 
    strlsm_n::{strlsm_left_notrans, strlsm_right_notrans},
    strlsm_t::{strlsm_left_trans,  strlsm_right_trans},
    strusm_n::{strusm_left_notrans, strusm_right_notrans},
    strusm_t::{strusm_left_trans,   strusm_right_trans},
};


/// Triangular matrix-matrix solve. 
/// `AX = alpha B` or `XA = alpha B` for `X` 
///
/// # Arguments 
/// * `side`: [CoralSide] - whether the LHS is `AX` (left) or `XA` (right)
/// * `uplo`: [CoralTriangular] - whether `A` is upper or lower triangular 
/// * `op_a`: [CoralTranspose] - whether `A` is transposed or not 
/// * `diag`: [CoralDiagonal] - whether `A` is unit-diagonal or not 
/// * `alpha`: [f32] - scalar for `alpha B` 
/// * `a`: [MatrixRef] - over [f32] 
/// * `b`: [MatrixMut] - over [f32], input as `B`, output as solved `X`. 
///
/// Returns: 
/// Nothing. `c` is updated in place with the solved matrix. 
#[inline(always)]
pub fn strsm(
    side: CoralSide,
    uplo: CoralTriangular,
    op_a: CoralTranspose,
    diag: CoralDiagonal,
    alpha: f32,
    a: MatrixRef<'_, f32>,
    mut b: MatrixMut<'_, f32>,
) {
    let a_t = op_a.is_transpose();
    let unit_diag = diag.is_unit();

    let (a_rows, a_cols) = (a.n_rows(), a.n_cols());
    let (b_rows, b_cols) = (b.n_rows(), b.n_cols());

    let m = b_rows;
    let n = b_cols;

    let (m_a_eff, n_a_eff) = if !a_t {
        (a_rows, a_cols)
    } else {
        (a_cols, a_rows)
    };

    match side {
        CoralSide::Left => {
            debug_assert_eq!(
                m_a_eff, m,
                "strsm: op(A) rows must match rows of C on left side."
            );
            debug_assert_eq!(
                m_a_eff, n_a_eff,
                "strsm: A must be square on left side."
            );
        }
        CoralSide::Right => {
            debug_assert_eq!(
                n_a_eff, n,
                "strsm: op(A) cols must match cols of C on right side."
            );
            debug_assert_eq!(
                m_a_eff, n_a_eff,
                "strsm: A must be square on right side."
            );
        }
    }

    let lda = a.lda();
    let ldc = b.lda();

    let a_data = a.as_slice();
    let b_data = b.as_slice_mut();

    let lda = a.lda();
    let ldc = b.lda();

    let a_data = a.as_slice();
    let c_data = b.as_slice_mut();

    match side {
        CoralSide::Left => {
            // one scratch buffer per left TRSM call
            // to avoid aliasing 
            let scratch_len = MC * n;
            let mut scratch = vec![0.0; scratch_len];

            match (uplo, a_t) {
                (CoralTriangular::Lower, false) => {
                    strlsm_left_notrans(
                        m, n, alpha, unit_diag,
                        a_data, lda,
                        c_data, ldc,
                        &mut scratch,
                    );
                }
                (CoralTriangular::Lower, true) => {
                    strlsm_left_trans(
                        m, n, alpha, unit_diag,
                        a_data, lda,
                        c_data, ldc,
                        &mut scratch,
                    );
                }
                (CoralTriangular::Upper, false) => {
                    strusm_left_notrans(
                        m, n, alpha, unit_diag,
                        a_data, lda,
                        c_data, ldc,
                        &mut scratch,
                    );
                }
                (CoralTriangular::Upper, true) => {
                    strusm_left_trans(
                        m, n, alpha, unit_diag,
                        a_data, lda,
                        c_data, ldc,
                        &mut scratch,
                    );
                }
            }
        }

        CoralSide::Right => {
            match (uplo, a_t) {
                (CoralTriangular::Lower, false) => {
                    strlsm_right_notrans(
                        m, n, alpha, unit_diag,
                        a_data, lda,
                        c_data, ldc,
                    );
                }
                (CoralTriangular::Lower, true) => {
                    strlsm_right_trans(
                        m, n, alpha, unit_diag,
                        a_data, lda,
                        c_data, ldc,
                    );
                }
                (CoralTriangular::Upper, false) => {
                    strusm_right_notrans(
                        m, n, alpha, unit_diag,
                        a_data, lda,
                        c_data, ldc,
                    );
                }
                (CoralTriangular::Upper, true) => {
                    strusm_right_trans(
                        m, n, alpha, unit_diag,
                        a_data, lda,
                        c_data, ldc,
                    );
                }
            }
        }
    }
}
