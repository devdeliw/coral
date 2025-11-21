use crate::types::{CoralDiagonal, CoralTranspose, MatrixRef, VectorRef, VectorMut};
use crate::fused::{saxpyf, sdotf};

const NB: usize = 8;


/// Solve NB x NB diagonal block for
/// lower-triangular no transpose A
#[inline]
fn forward_block_contiguous(
    nb: usize,
    unit_diag: bool,
    n: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    x: &mut [f32],
) {
    if nb == 0 { return; }

    debug_assert!(diag_idx + nb <= n);

    for li in 0..nb {
        let i = diag_idx + li;
        let mut sum = 0.0;

        for lk in 0..li {
            let j = diag_idx + lk;
            let a_ij = a[i + j * lda];
            let xj   = x[j];
            sum += a_ij * xj;
        }

        let mut xi = x[i] - sum;

        if !unit_diag {
            let a_ii = a[i + i * lda];
            xi /= a_ii;
        }

        x[i] = xi;
    }
}

/// Full forward substitution for lower-triangular 
/// no transpose A for generic incx
#[inline]
fn forward_full(
    n: usize,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    if n == 0 { return; }

    let step = incx;
    for i in 0..n {
        let mut sum = 0.0;

        for j in 0..i {
            let a_ij = a[i + j * lda]; 
            let xj   = x[j * step];
            sum += a_ij * xj;
        }

        let idx_xi = i * step;
        let mut xi = x[idx_xi] - sum;

        if !unit_diag {
            let a_ii = a[i + i * lda];
            xi /= a_ii;
        }

        x[idx_xi] = xi;
    }
}

/// Solve NB x NB diagonal block for 
/// lower-triangular transpoe A 
#[inline]
fn backward_block_contiguous(
    nb: usize,
    unit_diag: bool,
    n: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    x: &mut [f32],
) {
    if nb == 0 { return; }

    debug_assert!(diag_idx + nb <= n);

    // local li = 0..nb-1 ↔ global row/col i = diag_idx + li
    for li in (0..nb).rev() {
        let i = diag_idx + li;
        let mut sum = 0.0;

        // sum over k in block with k > i:
        // (L^T)[i, k] = L[k, i]
        for lk in (li + 1)..nb {
            let k  = diag_idx + lk;
            let a_ki = a[k + i * lda]; // L[k, i]
            let xk   = x[k];
            sum += a_ki * xk;
        }

        let mut xi = x[i] - sum;

        if !unit_diag {
            let a_ii = a[i + i * lda];
            xi /= a_ii;
        }

        x[i] = xi;
    }
}

/// Full backward substitution for lower-triangular 
/// transpose A for generic incx 
#[inline]
fn backward_full(
    n: usize,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    if n == 0 { return; }

    let step = incx;
    for i in (0..n).rev() {
        let mut sum = 0.0;

        // (L^T)[i, k] = L[k, i], k > i
        for k in (i + 1)..n {
            let a_ki = a[k + i * lda];
            let xk   = x[k * step];
            sum += a_ki * xk;
        }

        let idx_xi = i * step;
        let mut xi = x[idx_xi] - sum;

        if !unit_diag {
            let a_ii = a[i + i * lda];
            xi /= a_ii;
        }

        x[idx_xi] = xi;
    }
}


#[inline]
fn update_tail_notrans(
    rows_below: usize,
    nb: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    next_idx: usize,
    x: &mut [f32],
) {
    if rows_below == 0 || nb == 0 { return; }

    let a_panel_off = next_idx + diag_idx * lda;
    let a_panel_len = (nb - 1) * lda + rows_below;
    let a_panel     = &a[a_panel_off .. a_panel_off + a_panel_len];

    let x_block = &x[diag_idx .. diag_idx + nb];

    let mut x_block_neg = [0.0f32; NB];
    x_block_neg[..nb].copy_from_slice(&x_block[..nb]);
    for k in 0..nb {
        x_block_neg[k] = -x_block_neg[k];
    }

    let y_tail = &mut x[next_idx .. next_idx + rows_below];

    let abuf = MatrixRef::new(a_panel, rows_below, nb, lda, 0)
        .expect("A panel view failed");
    let xbuf = VectorRef::new(&x_block_neg[..nb], nb, 1, 0)
        .expect("x_block view failed");
    let ybuf = VectorMut::new(y_tail, rows_below, 1, 0)
        .expect("tail view failed");

    saxpyf(abuf, xbuf, ybuf);
}


#[inline]
fn update_head_transpose(
    head_len: usize,
    nb: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    x: &mut [f32],
) {
    if head_len == 0 || nb == 0 { return; }

    let a_view_off = diag_idx; 
    let a_view_len = (head_len - 1) * lda + nb;
    let a_view     = &a[a_view_off .. a_view_off + a_view_len];

    let x_block = &x[diag_idx .. diag_idx + nb];

    let mut x_block_neg = [0.0f32; NB];
    x_block_neg[..nb].copy_from_slice(&x_block[..nb]);
    for k in 0..nb {
        x_block_neg[k] = -x_block_neg[k];
    }

    let x_head = &mut x[..head_len];

    let abuf = MatrixRef::new(a_view, nb, head_len, lda, 0)
        .expect("A_left view failed");
    let xbuf = VectorRef::new(&x_block_neg[..nb], nb, 1, 0)
        .expect("x_block view failed");
    let ybuf = VectorMut::new(x_head, head_len, 1, 0)
        .expect("head view failed");

    sdotf(abuf, xbuf, ybuf);
}



#[inline]
fn strlsv_lower_notrans(
    n: usize,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    if n == 0 { return; }

    if incx == 1 {
        let nb      = NB;
        let nb_tail = n % nb;

        if n >= nb {
            let mut diag_idx = 0;

            while diag_idx + nb <= n {
                forward_block_contiguous(nb, unit_diag, n, a, lda, diag_idx, x);

                let next_idx = diag_idx + nb;
                if next_idx < n {
                    let rows_below = n - next_idx;
                    update_tail_notrans(rows_below, nb, a, lda, diag_idx, next_idx, x);
                }

                diag_idx += nb;
            }
        }

        if nb_tail > 0 {
            let idx = n - nb_tail;
            forward_block_contiguous(nb_tail, unit_diag, n, a, lda, idx, x);
        }
    } else {
        forward_full(n, unit_diag, a, lda, x, incx);
    }
}


#[inline]
fn strlsv_lower_trans(
    n: usize,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    if n == 0 { return; }

    if incx == 1 {
        let nb      = NB;
        let nb_tail = n % nb;

        if n >= nb {
            let mut diag_idx = n - nb;
            loop {
                backward_block_contiguous(nb, unit_diag, n, a, lda, diag_idx, x);

                if diag_idx > 0 {
                    let head_len = diag_idx;
                    update_head_transpose(head_len, nb, a, lda, diag_idx, x);
                }

                if diag_idx >= nb {
                    diag_idx -= nb;
                } else {
                    break;
                }
            }
        }

        if nb_tail > 0 {
            backward_block_contiguous(nb_tail, unit_diag, n, a, lda, 0, x);
        }
    } else {
        backward_full(n, unit_diag, a, lda, x, incx);
    }
}


#[inline]
pub(crate) fn strlsv(
    trans: CoralTranspose,
    diag: CoralDiagonal,
    a: MatrixRef<'_, f32>,
    mut x: VectorMut<'_, f32>,
) {
    let unit_diag = diag.is_unit(); 
    assert!(a.compare_m_n(), "n_cols must equal n_rows");

    let n    = a.n_rows();
    let lda  = a.lda();
    let abuf = a.as_slice();
    let incx = x.stride();
    let xbuf = x.as_slice_mut();

    match trans {
        CoralTranspose::NoTrans => strlsv_lower_notrans(n, unit_diag, abuf, lda, xbuf, incx),
        CoralTranspose::Trans   => strlsv_lower_trans(n, unit_diag, abuf, lda, xbuf, incx),
    }
}

