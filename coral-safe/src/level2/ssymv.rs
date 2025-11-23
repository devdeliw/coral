//! Level 2 [`?SYMV`](https://www.netlib.org/lapack/explore-html/dc/d0a/group__symv.html)
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


use std::simd::{Simd, StdFloat};
use std::simd::num::SimdFloat;

use crate::level2::pack_vector::pack_vector_f32;
use crate::types::{MatrixRef, VectorRef, VectorMut, CoralTriangular};

const LANES: usize = 8;


#[inline(always)]
fn upper_kernel(
    j: usize,
    temp1: f32,
    col: &[f32],
    xbuf: &[f32],
    ybuf: &mut [f32],
) -> f32 {
    if j == 0 {
        return 0.0;
    }

    let a_top = &col[..j];
    let x_top = &xbuf[..j];
    let y_top = &mut ybuf[..j];

    let (a_chunks, a_tail) = a_top.as_chunks::<LANES>();
    let (x_chunks, x_tail) = x_top.as_chunks::<LANES>();
    let (y_chunks, y_tail) = y_top.as_chunks_mut::<LANES>();

    debug_assert_eq!(a_chunks.len(), x_chunks.len());
    debug_assert_eq!(a_chunks.len(), y_chunks.len());

    let mut acc0 = Simd::<f32, LANES>::splat(0.0);
    let mut acc1 = Simd::<f32, LANES>::splat(0.0);
    let alpha_simd = Simd::<f32, LANES>::splat(temp1);

    let mut k = 0;
    let n_chunks = a_chunks.len();

    while k + 1 < n_chunks {
        // chunk k
        {
            let a_arr = a_chunks[k];
            let x_arr = x_chunks[k];
            let mut y_arr = y_chunks[k];

            let a = Simd::<f32, LANES>::from_array(a_arr);
            let x = Simd::<f32, LANES>::from_array(x_arr);
            let mut y = Simd::<f32, LANES>::from_array(y_arr);

            // y[i] += temp1 * a[i,j]
            y = a.mul_add(alpha_simd, y);
            y_arr = y.to_array();
            y_chunks[k] = y_arr;

            // temp2 += a[i,j] * x[i]
            acc0 = a.mul_add(x, acc0);
        }

        // chunk k + 1
        {
            let a_arr = a_chunks[k + 1];
            let x_arr = x_chunks[k + 1];
            let mut y_arr = y_chunks[k + 1];

            let a = Simd::<f32, LANES>::from_array(a_arr);
            let x = Simd::<f32, LANES>::from_array(x_arr);
            let mut y = Simd::<f32, LANES>::from_array(y_arr);

            y = a.mul_add(alpha_simd, y);
            y_arr = y.to_array();
            y_chunks[k + 1] = y_arr;

            acc1 = a.mul_add(x, acc1);
        }

        k += 2;
    }

    // leftover chunk
    if k < n_chunks {
        let a_arr = a_chunks[k];
        let x_arr = x_chunks[k];
        let mut y_arr = y_chunks[k];

        let a = Simd::<f32, LANES>::from_array(a_arr);
        let x = Simd::<f32, LANES>::from_array(x_arr);
        let mut y = Simd::<f32, LANES>::from_array(y_arr);

        y = a.mul_add(alpha_simd, y);
        y_arr = y.to_array();
        y_chunks[k] = y_arr;

        acc0 = a.mul_add(x, acc0);
    }

    let mut temp2 = (acc0 + acc1).reduce_sum();

    // tail
    for ((&aij, &xi), yi) in a_tail.iter().zip(x_tail.iter()).zip(y_tail.iter_mut()) {
        *yi += temp1 * aij;
        temp2 += aij * xi;
    }

    temp2
}

#[inline(always)]
fn lower_kernel(
    j: usize,
    n: usize,
    temp1: f32,
    col: &[f32],
    xbuf: &[f32],
    ybuf: &mut [f32],
) -> f32 {
    let start = j + 1;
    if start >= n {
        return 0.0;
    }

    let a_bot = &col[start..n];
    let x_bot = &xbuf[start..n];
    let y_bot = &mut ybuf[start..n];

    let (a_chunks, a_tail) = a_bot.as_chunks::<LANES>();
    let (x_chunks, x_tail) = x_bot.as_chunks::<LANES>();
    let (y_chunks, y_tail) = y_bot.as_chunks_mut::<LANES>();

    debug_assert_eq!(a_chunks.len(), x_chunks.len());
    debug_assert_eq!(a_chunks.len(), y_chunks.len());

    let mut acc0 = Simd::<f32, LANES>::splat(0.0);
    let mut acc1 = Simd::<f32, LANES>::splat(0.0);
    let alpha_simd = Simd::<f32, LANES>::splat(temp1);

    let mut k = 0;
    let n_chunks = a_chunks.len();

    while k + 1 < n_chunks {
        // chunk k
        {
            let a_arr = a_chunks[k];
            let x_arr = x_chunks[k];
            let mut y_arr = y_chunks[k];

            let a = Simd::<f32, LANES>::from_array(a_arr);
            let x = Simd::<f32, LANES>::from_array(x_arr);
            let mut y = Simd::<f32, LANES>::from_array(y_arr);

            // y[i] += temp1 * a[i,j]
            y = a.mul_add(alpha_simd, y);
            y_arr = y.to_array();
            y_chunks[k] = y_arr;

            // temp2 += a[i,j] * x[i]
            acc0 = a.mul_add(x, acc0);
        }

        // chunk k + 1
        {
            let a_arr = a_chunks[k + 1];
            let x_arr = x_chunks[k + 1];
            let mut y_arr = y_chunks[k + 1];

            let a = Simd::<f32, LANES>::from_array(a_arr);
            let x = Simd::<f32, LANES>::from_array(x_arr);
            let mut y = Simd::<f32, LANES>::from_array(y_arr);

            y = a.mul_add(alpha_simd, y);
            y_arr = y.to_array();
            y_chunks[k + 1] = y_arr;

            acc1 = a.mul_add(x, acc1);
        }

        k += 2;
    }

    // leftover chunk
    if k < n_chunks {
        let a_arr = a_chunks[k];
        let x_arr = x_chunks[k];
        let mut y_arr = y_chunks[k];

        let a = Simd::<f32, LANES>::from_array(a_arr);
        let x = Simd::<f32, LANES>::from_array(x_arr);
        let mut y = Simd::<f32, LANES>::from_array(y_arr);

        y = a.mul_add(alpha_simd, y);
        y_arr = y.to_array();
        y_chunks[k] = y_arr;

        acc0 = a.mul_add(x, acc0);
    }

    let mut temp2 = (acc0 + acc1).reduce_sum();

    // tail
    for ((&aij, &xi), yi) in a_tail.iter().zip(x_tail.iter()).zip(y_tail.iter_mut()) {
        *yi += temp1 * aij;
        temp2 += aij * xi;
    }

    temp2
}


#[inline]
fn upper (
    alpha: f32,
    beta: f32,
    a: MatrixRef<'_, f32>,
    x: VectorRef<'_, f32>,
    mut y: VectorMut<'_, f32>,
) {
    debug_assert!(a.compare_m_n(), "n_cols must equal n_rows");

    let n = a.n_rows();
    debug_assert!(x.compare_n(n), "logical length of x must equal n");
    debug_assert!(y.compare_n(n), "logical length of y must equal n");

    if n == 0 {
        return;
    }

    let incy = y.stride();
    if alpha == 0.0 {
        for yval in y.as_slice_mut().iter_mut().step_by(incy).take(n) { 
            *yval *= beta; 
        }
        return;
    }

    let incx = x.stride();
    let xoff = x.offset();
    let yoff = y.offset();

    // pack and scale into contiguous buffers
    let xdata = &x.as_slice()[xoff..];
    let ydata = &y.as_slice()[yoff..];
    let mut xbuf = Vec::new();
    let mut ybuf = Vec::new();
    pack_vector_f32(alpha, n, xdata, incx, &mut xbuf);
    if beta == 0.0 {
        ybuf.resize(n, 0.0);
    } else {
        pack_vector_f32(beta, n, ydata, incy, &mut ybuf);
    }

    let lda = a.lda();
    let aoff = a.offset();
    let aslice = a.as_slice();

    for j in 0..n {
        let temp1 = xbuf[j];

        let col_beg = aoff + j * lda;
        let col = &aslice[col_beg .. col_beg + n];

        let temp2 = upper_kernel(j, temp1, col, &xbuf, &mut ybuf);

        let ajj = col[j];
        ybuf[j] += temp1 * ajj + temp2;
    }

    if incy == 1 {
        let ys = &mut y.as_slice_mut()[yoff .. yoff + n];
        ys.copy_from_slice(&ybuf[..n]);
    } else {
        let ys = y.as_slice_mut();
        let ys_it = ys[yoff..]
            .iter_mut()
            .step_by(incy)
            .take(n);

        for (ynew, &yold) in ys_it.zip(ybuf.iter()) {
            *ynew = yold;
        }
    }
}

#[inline]
fn lower (
    alpha: f32,
    beta: f32,
    a: MatrixRef<'_, f32>,
    x: VectorRef<'_, f32>,
    mut y: VectorMut<'_, f32>,
) {
    debug_assert!(a.compare_m_n(), "n_cols must equal n_rows");

    let n = a.n_rows();
    debug_assert!(x.compare_n(n), "logical length of x must equal n");
    debug_assert!(y.compare_n(n), "logical length of y must equal n");

    if n == 0 {
        return;
    }

    let incy = y.stride(); 
    if alpha == 0.0 {
        for yval in y.as_slice_mut().iter_mut().step_by(incy).take(n) { 
            *yval *= beta; 
        }
        return;
    }

    let incx = x.stride();
    let xoff = x.offset();
    let yoff = y.offset();

    // pack and scale into contiguous buffers 
    let xdata = &x.as_slice()[xoff..];
    let ydata = &y.as_slice()[yoff..];
    let mut xbuf = Vec::new();
    let mut ybuf = Vec::new();
    pack_vector_f32(alpha, n, xdata, incx, &mut xbuf);
    if beta == 0.0 {
        ybuf.resize(n, 0.0);
    } else {
        pack_vector_f32(beta, n, ydata, incy, &mut ybuf);
    }

    let lda = a.lda();
    let aoff = a.offset();
    let aslice = a.as_slice();

    for j in 0..n {
        let temp1 = xbuf[j];

        let col_beg = aoff + j * lda;
        let col = &aslice[col_beg .. col_beg + n];

        let ajj = col[j];
        ybuf[j] += temp1 * ajj;

        let temp2 = lower_kernel(j, n, temp1, col, &xbuf, &mut ybuf);
        ybuf[j] += temp2;
    }

    if incy == 1 {
        let ys = &mut y.as_slice_mut()[yoff .. yoff + n];
        ys.copy_from_slice(&ybuf[..n]);
    } else {
        let ys = y.as_slice_mut();
        let ys_it = ys[yoff..]
            .iter_mut()
            .step_by(incy)
            .take(n);

        for (ynew, &yold) in ys_it.zip(ybuf.iter()) {
            *ynew = yold;
        }
    }
}


/// Performs symmetric matrix-vector multiply. 
///
/// Arguments: 
/// * `uplo`: [CoralTriangular] - indicates which triangle to use 
/// * `alpha`: [f32] - scalar for `alpha * A x `
/// * `beta`: [f32] - scalar for `beta y` 
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
    y: VectorMut<'_, f32>,
) {
    match uplo {
        CoralTriangular::Upper => upper(alpha, beta, a, x, y),
        CoralTriangular::Lower => lower(alpha, beta, a, x, y),
    }
}

