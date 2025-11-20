/// Packs a rectangular panel of columns from a matrix into a contiguous 
/// column-major buffer. 
#[inline] 
pub(crate) fn pack_panel<T: Copy> ( 
    panel: &mut Vec<T>, 
    matrix: &[T], 
    n_rows: usize, 
    n_cols: usize, 
    col_idx: usize, 
    lda: usize
) { 
    let total = n_rows * n_cols; 
    panel.clear(); 
    panel.reserve_exact(total); 

    for col in 0..n_cols { 
        let col_beg = (col_idx + col) * lda; 
        let col_end = col_beg + n_rows; 
        let col_slice = &matrix[col_beg .. col_end];

        panel.extend_from_slice(col_slice); 
    }
}
