//! Level 2 [`?SYMV`](https://www.netlib.org/lapack/explore-html/db/d17/group__hemv_ga8990fe737209f3401522103c85016d27.html)
//! routine in single precision. 
//!
//! \\[ 
//! y \leftarrow \alpha A x + \beta y 
//! \\]
//!
//! where $A$ symmetric $n \times n$ matrix. 
//!
//! # Author 
//! Deval Deliwala


use crate::types::{MatrixRef, VectorMut, VectorRef, CoralTriangular};
use crate::fused::{saxpyf, sdotf};
use crate::level2::pack_panel::pack_panel;
use crate::level2::pack_vector::pack_vector_f32;

const MC: usize = 8;
const NC: usize = 32;


/// Uses upper-triangle of symmetric matrix
/// `lda == n` fast path
#[inline]
fn upper(
    n: usize,
    a: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    for j in 0..n {
        let xj = x[j];
        let mut acc = 0.0;
        let col_off = j * n;

        for i in 0..j {
            let a_ij = a[col_off + i];

            // y[i] += a_ij * xj
            y[i] = a_ij.mul_add(xj, y[i]);
            acc = a_ij.mul_add(x[i], acc);
        }

        // diagonal
        let a_jj = a[col_off + j];
        y[j] = a_jj.mul_add(xj, y[j]);
        y[j] += acc;
    }
}

/// Uses lower-triangle of symmetric matrix
/// `lda == n` fast path
#[inline]
fn lower(
    n: usize,
    a: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    for j in 0..n {
        let xj = x[j];
        let mut acc = 0.0;
        let col_off = j * n;

        let mut i = j + 1;
        while i < n {
            let a_ij = a[col_off + i];

            // y[i] += a_ij * xj
            y[i] = a_ij.mul_add(xj, y[i]);
            acc = a_ij.mul_add(x[i], acc);
            i += 1;
        }

        // diagonal
        let a_jj = a[col_off + j];
        y[j] = a_jj.mul_add(xj, y[j]);
        y[j] += acc;
    }
}

/// Uses upper-triangle of symmetric matrix
/// for `lda > n` slow path
#[inline]
fn fallback_upper(
    n: usize,
    lda: usize,
    matrix: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    let mut apack: Vec<f32> = Vec::new();

    let mut row_idx = 0;
    while row_idx < n {
        let mb = core::cmp::min(MC, n - row_idx);

        // diagonal block to `mb`
        {
            let x_head = &x[row_idx .. row_idx + mb];
            let y_head = &mut y[row_idx .. row_idx + mb];

            for j in 0..mb {
                let col_abs = row_idx + j;
                let col_off = col_abs * lda;
                let xj = x_head[j];
                let mut acc = 0.0;

                for i in 0..j {
                    let row_abs = row_idx + i;
                    let a_ij = matrix[col_off + row_abs];

                    // y_head[i] += a_ij * xj
                    y_head[i] = a_ij.mul_add(xj, y_head[i]);
                    acc = a_ij.mul_add(x_head[i], acc);
                }

                // diagonal
                let diag_row = col_abs;
                let a_jj = matrix[col_off + diag_row];
                y_head[j] = a_jj.mul_add(xj, y_head[j]);
                y_head[j] += acc;
            }
        }

        // off-diagonal
        {
            let a_row_base = &matrix[row_idx ..];

            let mut col_idx = row_idx + mb;
            while col_idx < n {
                let nb = core::cmp::min(NC, n - col_idx);

                pack_panel(
                    &mut apack,
                    a_row_base,
                    mb,
                    nb,
                    col_idx,
                    lda,
                );
                let panel = &apack[..];

                let (y_pre, y_post) = y.split_at_mut(row_idx + mb);
                let y_head: &mut [f32] = &mut y_pre[row_idx ..]; // len = mb
                let start_tail = col_idx - (row_idx + mb);
                let y_tail: &mut [f32] =
                    &mut y_post[start_tail .. start_tail + nb];

                let a_panel = MatrixRef::new(
                    panel,
                    mb,
                    nb,
                    mb,
                    0,
                )
                .expect("ssymv(fallback_upper): panel matrix construction failed");

                let x_tail = &x[col_idx .. col_idx + nb];
                let x_tail_v = VectorRef::new(
                    x_tail,
                    nb,
                    1,
                    0,
                )
                .expect("ssymv(fallback_upper): x_tail vector construction failed");

                let y_head_v = VectorMut::new(
                    y_head,
                    mb,
                    1,
                    0,
                )
                .expect("ssymv(fallback_upper): y_head vector construction failed");

                saxpyf(a_panel, x_tail_v, y_head_v);

                let a_panel = MatrixRef::new(
                    panel,
                    mb,
                    nb,
                    mb,
                    0,
                )
                .expect("ssymv(fallback_upper): panel matrix construction failed");

                let x_head = &x[row_idx .. row_idx + mb];
                let x_head_v = VectorRef::new(
                    x_head,
                    mb,
                    1,
                    0,
                )
                .expect("ssymv(fallback_upper): x_head vector construction failed");

                let y_tail_v = VectorMut::new(
                    y_tail,
                    nb,
                    1,
                    0,
                )
                .expect("ssymv(fallback_upper): y_tail vector construction failed");

                sdotf(a_panel, x_head_v, y_tail_v);

                col_idx += nb;
            }
        }

        row_idx += mb;
    }
}

/// Uses lower-triangle of symmetric matrix
/// for `lda > n` slow path 
#[inline]
fn fallback_lower(
    n: usize,
    lda: usize,
    matrix: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    let mut apack: Vec<f32> = Vec::new();

    let mut row_idx = 0;
    while row_idx < n {
        let mb = core::cmp::min(MC, n - row_idx);

        // off-diagonal 
        {
            let a_row_base = &matrix[row_idx ..];

            let mut col_idx = 0;
            while col_idx < row_idx {
                let nb = core::cmp::min(NC, row_idx - col_idx);

                pack_panel(
                    &mut apack,
                    a_row_base,
                    mb,
                    nb,
                    col_idx,
                    lda,
                );
                let panel = &apack[..];

                let (y_left_region, y_head_region) = y.split_at_mut(row_idx);
                let y_left: &mut [f32] =
                    &mut y_left_region[col_idx .. col_idx + nb];
                let y_head: &mut [f32] =
                    &mut y_head_region[..mb];

                let a_panel = MatrixRef::new(
                    panel,
                    mb,
                    nb,
                    mb,
                    0,
                )
                .expect("ssymv(fallback_lower): panel matrix construction failed");

                let x_left = &x[col_idx .. col_idx + nb];
                let x_left_v = VectorRef::new(
                    x_left,
                    nb,
                    1,
                    0,
                )
                .expect("ssymv(fallback_lower): x_left vector construction failed");

                let y_head_v = VectorMut::new(
                    y_head,
                    mb,
                    1,
                    0,
                )
                .expect("ssymv(fallback_lower): y_head vector construction failed");

                saxpyf(a_panel, x_left_v, y_head_v);

                // y_left += panel^T * x_head
                let a_panel = MatrixRef::new(
                    panel,
                    mb,
                    nb,
                    mb,
                    0,
                )
                .expect("ssymv(fallback_lower): panel matrix construction failed");

                let x_head = &x[row_idx .. row_idx + mb];
                let x_head_v = VectorRef::new(
                    x_head,
                    mb,
                    1,
                    0,
                )
                .expect("ssymv(fallback_lower): x_head vector construction failed");

                let y_left_v = VectorMut::new(
                    y_left,
                    nb,
                    1,
                    0,
                )
                .expect("ssymv(fallback_lower): y_left vector construction failed");

                sdotf(a_panel, x_head_v, y_left_v);

                col_idx += nb;
            }
        }

        // diagonal to `mb` 
        {
            let x_head = &x[row_idx .. row_idx + mb];
            let y_head = &mut y[row_idx .. row_idx + mb];

            for j in 0..mb {
                let col_abs = row_idx + j;
                let col_off = col_abs * lda;
                let xj = x_head[j];
                let mut acc = 0.0;

                let mut i = j + 1;
                while i < mb {
                    let row_abs = row_idx + i;
                    let a_ij = matrix[col_off + row_abs];

                    // y_head[i] += a_ij * xj
                    y_head[i] = a_ij.mul_add(xj, y_head[i]);
                    acc = a_ij.mul_add(x_head[i], acc);
                    i += 1;
                }

                // diagonal
                let diag_row = col_abs;
                let a_jj = matrix[col_off + diag_row];
                y_head[j] = a_jj.mul_add(xj, y_head[j]);
                y_head[j] += acc;
            }
        }

        row_idx += mb;
    }
}


/// Performs a symmetric matrix-vector multiply. 
/// `y := alpha A x + beta y` 
///
/// Arguments: 
/// * `uplo`: [CoralTriangular] - which triangle of `A` to use 
/// * `alpha`: [f32] - scaling constant for `alpha A x` 
/// * `beta`: [f32] - scaling constant for `beta y` 
/// * `a`: [MatrixRef] - over [f32], symmetric `n x n` 
/// * `x`: [VectorRef] - over [f32]
/// * `y`: [VectorMut] - over [f32]
///
/// Returns: 
/// Nothing. `y.data` is updated in place. 
#[inline]
pub fn ssymv(
    uplo: CoralTriangular,
    alpha: f32,
    beta: f32,
    a: MatrixRef<'_, f32>,
    x: VectorRef<'_, f32>,
    mut y: VectorMut<'_, f32>,
) {
    let n = a.n_rows();
    assert!(a.compare_m_n(), "n_rows must equal n_cols");
    assert!(x.compare_n(n), "length of x must equal n symmetrix matrix");
    assert!(y.compare_n(n), "length of x must equal n symmetrix matrix");

    if n == 0 {
        return;
    }

    if alpha == 0.0 && beta == 1.0 {
        return;
    }

    let incx = x.stride();
    let incy = y.stride();
    let xoff = x.offset();
    let yoff = y.offset();
    let xdata = &x.as_slice()[xoff ..];
    let ydata = &y.as_slice()[yoff ..];

    let mut xbuf: Vec<f32> = Vec::new();
    // packs and scales `alpha * x` into contiguous buffer
    pack_vector_f32(alpha, n, xdata, incx, &mut xbuf);

    let mut ybuf: Vec<f32> = Vec::new();
    // packs and scales `beta * y` into contiguous buffer
    pack_vector_f32(beta, n, ydata, incy, &mut ybuf);

    let lda = a.lda();
    let aslice = a.as_slice();

    match uplo {
        CoralTriangular::Upper => {
            if lda == n {
                upper(n, aslice, &xbuf, &mut ybuf);
            } else {
                fallback_upper(n, lda, aslice, &xbuf, &mut ybuf);
            }
        }
        CoralTriangular::Lower => {
            if lda == n {
                lower(n, aslice, &xbuf, &mut ybuf);
            } else {
                fallback_lower(n, lda, aslice, &xbuf, &mut ybuf);
            }
        }
    }

    if incy == 1 {
        let ys = &mut y.as_slice_mut()[yoff .. yoff + n];
        ys.copy_from_slice(&ybuf[..n]);
    } else {
        let ys = y.as_slice_mut();
        for (dst, &src) in ys[yoff ..]
            .iter_mut()
            .step_by(incy)
            .take(n)
            .zip(ybuf.iter())
        {
            *dst = src;
        }
    }
}

