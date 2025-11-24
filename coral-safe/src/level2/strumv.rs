use crate::types::{CoralDiagonal, CoralTranspose, MatrixRef, VectorRef, VectorMut}; 
use crate::fused::{saxpyf, sdotf}; 

const NB_NOTRANS: usize = 32; 
const NB_TRANS: usize = 8; 


/// Parses contiguous `nb x nb` diagonal block 
/// for upper-triangular no transpose A
#[inline]
fn notrans_contiguous(
    nb: usize,
    unit_diag: bool,
    n: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    x: &mut [f32],
) {
    if nb == 0 {
        return;
    }

    debug_assert!(diag_idx + nb <= n);
    debug_assert!(nb <= NB_NOTRANS);

    let mut x_block = [0.0; NB_NOTRANS];
    x_block[..nb].copy_from_slice(&x[diag_idx .. diag_idx + nb]);

    let mut y_block = [0.0; NB_NOTRANS];

    for li in 0..nb {
        let i = diag_idx + li;

        let xi = x_block[li];
        let mut sum = if unit_diag {
            xi
        } else {
            let a_ii = a[i + i * lda];
            a_ii * xi
        };

        for (lj, xval) in x_block[(li + 1)..nb].iter().enumerate() {
            let j = diag_idx + lj + li + 1;
            let a_ij = a[i + j * lda];
            sum += a_ij * xval;
        }

        y_block[li] = sum;
    }

    x[diag_idx .. diag_idx + nb].copy_from_slice(&y_block[..nb]);
}

#[inline]
fn update_tail_notrans (
    cols_right: usize,
    nb: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    next_idx: usize,
    x: &mut [f32],
) {
    if cols_right == 0 || nb == 0 {
        return;
    } 

    debug_assert!(nb <= NB_NOTRANS);
    debug_assert_eq!(next_idx, diag_idx + nb);          

    let a_panel_off = diag_idx + next_idx * lda;
    let a_panel_len = (cols_right - 1) * lda + nb;
    let a_panel = &a[a_panel_off .. a_panel_off + a_panel_len];

    let (head, tail) = x.split_at_mut(next_idx);

    let y_block = &mut head[diag_idx .. diag_idx + nb];
    let x_right = &tail[..cols_right];

    let abuf = MatrixRef::new(a_panel, nb, cols_right, lda, 0)
        .expect("a view failed");
    let xbuf = VectorRef::new(x_right, cols_right, 1, 0)
        .expect("x view failed");
    let ybuf = VectorMut::new(y_block, nb, 1, 0)
        .expect("y view failed");

    saxpyf(abuf, xbuf, ybuf);
}

/// Full multiply for no transpose 
/// upper-triangular A for generic incx
#[inline]
fn notrans_full (
    n: usize,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    if n == 0 {
        return;
    }

    let step = incx;
    for i in 0..n {
        let idx_xi = i * step;
        let xi = x[idx_xi];

        let mut sum = if unit_diag {
            xi
        } else {
            let a_ii = a[i + i * lda];
            a_ii * xi
        };

        for j in (i + 1)..n {
            let xj = x[j * step];
            let a_ij = a[i + j * lda];
            sum += a_ij * xj;
        }

        x[idx_xi] = sum;
    }
}

/// Parses contiguous `nb x nb` contiguous 
/// block for upper-triangular transpose A 
#[inline]
fn trans_contiguous (
    nb: usize,
    unit_diag: bool,
    n: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    x: &mut [f32],
) {
    if nb == 0 {
        return;
    }

    debug_assert!(diag_idx + nb <= n);
    debug_assert!(nb <= NB_TRANS);

    let mut x_block = [0.0; NB_TRANS];
    x_block[..nb].copy_from_slice(&x[diag_idx .. diag_idx + nb]);

    let mut y_block = [0.0; NB_TRANS];

    for lk in 0..nb {
        let j = diag_idx + lk; 
        let col_offset = j * lda;

        let mut sum = 0.0;

        for (li, xval) in x_block[0..lk].iter().enumerate() { 
            let i = diag_idx + li;
            let a_ij = a[i + col_offset];
            sum += a_ij * xval;
        }

        if unit_diag {
            sum += x_block[lk];
        } else {
            let i_diag = diag_idx + lk;
            let a_kk = a[i_diag + col_offset];
            sum += a_kk * x_block[lk];
        }

        y_block[lk] = sum;
    }

    x[diag_idx .. diag_idx + nb].copy_from_slice(&y_block[..nb]);
}

#[inline]
fn update_head_trans (
    rows_above: usize,
    nb: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    x: &mut [f32],
) {
    if rows_above == 0 || nb == 0 {
        return;
    }

    debug_assert_eq!(rows_above, diag_idx);
    debug_assert!(nb <= NB_TRANS);

    let a_panel_off = diag_idx * lda; 
    let a_panel_len = (nb - 1) * lda + rows_above;
    let a_panel = &a[a_panel_off .. a_panel_off + a_panel_len];

    let (head, tail) = x.split_at_mut(diag_idx);

    let x_head  = &head[..rows_above];   
    let y_block = &mut tail[..nb];       

    let abuf = MatrixRef::new(a_panel, rows_above, nb, lda, 0)
        .expect("a view failed");
    let xbuf = VectorRef::new(x_head, rows_above, 1, 0)
        .expect("x view failed");
    let ybuf = VectorMut::new(y_block, nb, 1, 0)
        .expect("y view failed");

    sdotf(abuf, xbuf, ybuf);
}

/// Full multiply for transpose
/// upper-triangular A for generic incx 
#[inline]
fn trans_full (
    n: usize,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    if n == 0 {
        return;
    }

    let step = incx;
    for i in (0..n).rev() {
        let idx_xi = i * step;
        let xi = x[idx_xi];

        let mut sum = if unit_diag {
            xi
        } else {
            let a_ii = a[i + i * lda];
            a_ii * xi
        };

        for j in 0..i {
            let a_ji = a[j + i * lda]; 
            let xj = x[j * step];
            sum += a_ji * xj;
        }

        x[idx_xi] = sum;
    }
}

#[inline]
fn notrans (
    n: usize,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    if n == 0 {
        return;
    }

    if incx == 1 {
        let nb = NB_NOTRANS;
        let nb_tail = n % nb;

        if n >= nb {
            let mut diag_idx = 0;

            while diag_idx + nb <= n {
                notrans_contiguous(nb, unit_diag, n, a, lda, diag_idx, x);

                let next_idx = diag_idx + nb;
                if next_idx < n {
                    let cols_right = n - next_idx;
                    update_tail_notrans(cols_right, nb, a, lda, diag_idx, next_idx, x);
                }

                diag_idx += nb;
            }
        }

        if nb_tail > 0 {
            let diag_idx_tail = n - nb_tail;
            notrans_contiguous(nb_tail, unit_diag, n, a, lda, diag_idx_tail, x);
        }
    } else {
        notrans_full(n, unit_diag, a, lda, x, incx);
    }
}

#[inline]
fn trans (
    n: usize,
    unit_diag: bool,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    if n == 0 {
        return;
    }

    if incx == 1 {
        let nb = NB_TRANS;
        let nb_tail = n % nb;

        if n >= nb {
            let mut diag_idx = n;
            while diag_idx >= nb {
                diag_idx -= nb;
                trans_contiguous(nb, unit_diag, n, a, lda, diag_idx, x);

                if diag_idx > 0 {
                    let rows_above = diag_idx;
                    update_head_trans(rows_above, nb, a, lda, diag_idx, x);
                }
            }
        }

        if nb_tail > 0 {
            trans_contiguous(nb_tail, unit_diag, n, a, lda, 0, x);
        }
    } else {
        trans_full(n, unit_diag, a, lda, x, incx);
    }
}


#[inline]
pub(crate) fn strumv (
    transpose: CoralTranspose,
    diag: CoralDiagonal,
    a: MatrixRef<'_, f32>,
    mut x: VectorMut<'_, f32>,
) {
    let unit_diag = diag.is_unit();
    debug_assert!(a.compare_m_n(), "n_cols must equal n_rows");

    let n    = a.n_rows();
    let lda  = a.lda();
    let abuf = a.as_slice();
    let incx = x.stride();
    let xbuf = x.as_slice_mut();

    match transpose {
        CoralTranspose::NoTrans => notrans(n, unit_diag, abuf, lda, xbuf, incx),
        CoralTranspose::Trans   => trans(n, unit_diag, abuf, lda, xbuf, incx)
    }
}
