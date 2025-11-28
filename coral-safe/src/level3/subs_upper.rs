/// backwards substitution for Left NoTranspose STRUSM variant 
#[inline(always)]
pub(crate) fn backward_sub_panel_left_upper_n (
    m: usize,
    n: usize,
    unit_diag: bool,
    a_ii: &[f32],
    lda: usize,
    b_i: &mut [f32],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }

    for j in 0..n {
        let col_offs = j * ldb;
        let col = &mut b_i[col_offs .. col_offs + m];

        let mut i = m;
        while i > 0 {
            i -= 1;

            let mut tmp = col[i];

            let mut k = i + 1;
            while k < m {
                let a_ik_idx = i + k * lda;   
                tmp -= a_ii[a_ik_idx] * col[k];

                k += 1;
            }

            if !unit_diag {
                let diag = a_ii[i + i * lda];

                debug_assert!(diag != 0.0);
                tmp /= diag;
            }

            col[i] = tmp;
        }
    }
}

/// forwards substitution for Right NoTranspose STRUSM variant 
#[inline(always)]
pub(crate) fn forward_sub_panel_right_upper_n (
    m: usize,
    n: usize,
    unit_diag: bool,
    a_ii: &[f32],
    lda: usize,
    b_i: &mut [f32],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }

    for j in 0..n {
        for row in 0..m {
            let idx = row + j * ldb;
            let mut tmp = b_i[idx];

            let mut k = 0;
            while k < j {
                let x_k  = b_i[row + k * ldb];
                let u_kj = a_ii[k + j * lda];
                tmp -= x_k * u_kj;

                k += 1;
            }

            if !unit_diag {
                let diag = a_ii[j + j * lda];

                debug_assert!(diag != 0.0);
                tmp /= diag;
            }

            b_i[idx] = tmp;
        }
    }
}

/// forwards substitution for Left Transpose STRUSM variant 
#[inline(always)]
pub(crate) fn forward_sub_panel_left_upper_t (
    m: usize,
    n: usize,
    unit_diag: bool,
    a_ii: &[f32],
    lda: usize,
    b_i: &mut [f32],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }

    for j in 0..n {
        let col_offs = j * ldb;
        let col = &mut b_i[col_offs .. col_offs + m];

        for i in 0..m {
            if !unit_diag {
                let diag_idx = i + i * lda;
                let diag = a_ii[diag_idx];

                debug_assert!(diag != 0.0);
                col[i] /= diag;
            }

            let x_i = col[i];

            let mut k = i + 1;
            while k < m {
                let a_ik_idx = i + k * lda;   
                col[k] -= a_ii[a_ik_idx] * x_i;

                k += 1;
            }
        }
    }
} 

/// backwards substitution for Right Transpose STRUSM variant
#[inline(always)]
pub(crate) fn backward_sub_panel_right_upper_t (
    m: usize,
    n: usize,
    unit_diag: bool,
    a_ii: &[f32],
    lda: usize,
    b_i: &mut [f32],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }

    for row in 0..m {
        let mut j = n;
        while j > 0 {
            j -= 1;

            let idx = row + j * ldb;
            let mut tmp = b_i[idx];

            let mut k = j + 1;
            while k < n {
                let x_k  = b_i[row + k * ldb];
                let u_jk = a_ii[j + k * lda];   
                tmp -= x_k * u_jk;

                k += 1;
            }

            if !unit_diag {
                let diag = a_ii[j + j * lda]; 

                debug_assert!(diag != 0.0);
                tmp /= diag;
            }

            b_i[idx] = tmp;
        }
    }
}

