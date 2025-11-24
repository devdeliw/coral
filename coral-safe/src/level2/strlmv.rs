use crate::types::{CoralDiagonal, CoralTranspose, MatrixRef, VectorRef, VectorMut};
use crate::fused::{saxpyf, sdotf};

const NB_NOTRANS: usize = 32;
const NB_TRANS:   usize = 8;


/// Parses contiguouso `nb x nb` diagonal block 
/// for lower-triangular no transpose A 
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

        let mut sum = if unit_diag {
            x_block[li]
        } else {
            let a_ii = a[i + i * lda];
            a_ii * x_block[li]
        };

        for (lj, xval) in x_block[0..li].iter().enumerate() {
            let j = diag_idx + lj;
            let a_ij = a[i + j * lda];
            sum += a_ij * xval; 
        }

        y_block[li] = sum;
    }

    x[diag_idx .. diag_idx + nb].copy_from_slice(&y_block[..nb]);
}

#[inline]
fn update_head_notrans(
    cols_left: usize,
    nb: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    x: &mut [f32],
) {
    if cols_left == 0 || nb == 0 {
        return;
    }

    debug_assert_eq!(cols_left, diag_idx);
    debug_assert!(nb <= NB_NOTRANS);

    let a_panel_off = diag_idx;
    let a_panel_len = (cols_left - 1) * lda + nb;
    let a_panel = &a[a_panel_off .. a_panel_off + a_panel_len];

    let (head, tail) = x.split_at_mut(diag_idx);

    let x_left  = &head[..cols_left];   
    let y_block = &mut tail[..nb];      

    let abuf = MatrixRef::new(a_panel, nb, cols_left, lda, 0)
        .expect("strlmv: A panel view failed");
    let xbuf = VectorRef::new(x_left, cols_left, 1, 0)
        .expect("strlmv: x_left view failed");
    let ybuf = VectorMut::new(y_block, nb, 1, 0)
        .expect("strlmv: y_block view failed");

    saxpyf(abuf, xbuf, ybuf);
}


/// Full multiply for no transpose
/// lower-triangular A for generic incx 
#[inline]
fn notrans_full(
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
            let idx_xj = j * step;
            let xj = x[idx_xj];
            let a_ij = a[i + j * lda];
            sum += a_ij * xj;
        }

        x[idx_xi] = sum;
    }
}


/// Parses contiguous `nb x nb` diagonal block 
/// for transpose lower-triangular A 
#[inline]
fn trans_contiguous(
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
        let col = diag_idx + lk;
        let col_offset = col * lda;

        let mut sum = 0.0;

        for (li, xval) in x_block[(lk + 1)..nb].iter().enumerate() {
            let row = diag_idx + li + lk + 1;
            let a_ij = a[row + col_offset];
            sum += a_ij * xval;
        }

        if unit_diag {
            sum += x_block[lk];
        } else {
            let a_jj = a[col + col_offset];
            sum += a_jj * x_block[lk];
        }

        y_block[lk] = sum;
    }

    x[diag_idx .. diag_idx + nb].copy_from_slice(&y_block[..nb]);
}

#[inline]
fn update_tail_trans(
    rows_below: usize,
    nb: usize,
    a: &[f32],
    lda: usize,
    diag_idx: usize,
    n: usize,
    x: &mut [f32],
) {
    if rows_below == 0 || nb == 0 {
        return;
    }

    let row_tail = diag_idx + nb;
    debug_assert_eq!(rows_below, n - row_tail);
    debug_assert!(nb <= NB_TRANS);

    let a_panel_off = row_tail + diag_idx * lda;
    let a_panel_len = (nb - 1) * lda + rows_below;
    let a_panel = &a[a_panel_off .. a_panel_off + a_panel_len];

    let (head, tail) = x.split_at_mut(row_tail);

    let x_tail  = &tail[..rows_below];              
    let y_block = &mut head[diag_idx .. diag_idx + nb]; 

    let abuf = MatrixRef::new(a_panel, rows_below, nb, lda, 0)
        .expect("strlmv: A panel (below) view failed");
    let xbuf = VectorRef::new(x_tail, rows_below, 1, 0)
        .expect("strlmv: x_tail view failed");
    let ybuf = VectorMut::new(y_block, nb, 1, 0)
        .expect("strlmv: y_block view failed");

    sdotf(abuf, xbuf, ybuf);
}


/// Full multiply for transpose 
/// lower-triangular A for generic incx 
#[inline]
fn trans_full(
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

    for j in 0..n {
        let idx_xj = j * step;
        let xj = x[idx_xj];

        let mut sum = if unit_diag {
            xj
        } else {
            let a_jj = a[j + j * lda];
            a_jj * xj
        };

        for i in (j + 1)..n {
            let idx_xi = i * step;
            let xi = x[idx_xi];
            let a_ij = a[i + j * lda]; 
            sum += a_ij * xi;
        }

        x[idx_xj] = sum;
    }
}

#[inline]
fn notrans(
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
            let mut diag_idx = n;
            while diag_idx >= nb {
                diag_idx -= nb;

                notrans_contiguous(nb, unit_diag, n, a, lda, diag_idx, x);

                if diag_idx > 0 {
                    let cols_left = diag_idx;
                    update_head_notrans(cols_left, nb, a, lda, diag_idx, x);
                }
            }
        }

        if nb_tail > 0 {
            let diag_idx_tail = 0;
            notrans_contiguous(nb_tail, unit_diag, n, a, lda, diag_idx_tail, x);
        }
    } else {
        notrans_full(n, unit_diag, a, lda, x, incx);
    }
}

#[inline]
fn trans(
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
            let mut diag_idx = 0;
            while diag_idx + nb <= n {
                trans_contiguous(nb, unit_diag, n, a, lda, diag_idx, x);

                let row_tail = diag_idx + nb;
                if row_tail < n {
                    let rows_below = n - row_tail;
                    update_tail_trans(rows_below, nb, a, lda, diag_idx, n, x);
                }

                diag_idx += nb;
            }
        }

        if nb_tail > 0 {
            let diag_idx_tail = n - nb_tail;
            trans_contiguous(nb_tail, unit_diag, n, a, lda, diag_idx_tail, x);
        }
    } else {
        trans_full(n, unit_diag, a, lda, x, incx);
    }
}


#[inline]
pub(crate) fn strlmv(
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
        CoralTranspose::Trans   => trans  (n, unit_diag, abuf, lda, xbuf, incx),
    }
}

