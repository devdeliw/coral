use rand::thread_rng;
use rand::distributions::{Distribution, Standard}; 
use coral_safe::types::{VectorMut, VectorRef, MatrixMut, MatrixRef};

/// Make a `Vec<f32>` buffer of length `len` 
/// with stride `inc` with randomized elems
#[inline]
pub(crate) fn make_strided_vec (
    len: usize, 
    inc: usize
) -> Vec<f32> {

    if len == 0 { 
        return vec![1.0; 1]; 
    }

    let req_len = (len - 1) * inc + (len > 0) as usize;
    let mut buf = vec![0.0; req_len];

    let mut rng = thread_rng();
    let dist = Standard;
    for val in buf.iter_mut().step_by(inc).take(len) {
        *val = dist.sample(&mut rng);
    }

    buf
}


#[inline] 
pub(crate) fn make_strided_mat (
    n_rows: usize,
    n_cols: usize, 
    lda: usize, 
) -> Vec<f32> {
    if lda == 0 || n_cols == 0 { 
        return vec![1.0; 1]; 
    }

    let req_len = lda * n_cols; 
    let mut buf = vec![0.0; req_len]; 

    let mut rng = thread_rng(); 
    let dist = Standard; 

    for m in 0..n_rows {  
        for j in 0..n_cols { 
            buf[m + j * lda] = dist.sample(&mut rng); 
        }
    }

    buf 
}


#[inline]
pub(crate) fn bytes(n: usize, alpha: usize) -> u64 { 
    (alpha * n * std::mem::size_of::<f32>()) as u64
}

#[inline] 
pub(crate) fn make_view_ref<'a>(x: &'a [f32], n: usize, incx: usize) -> VectorRef<'a, f32> { 
    VectorRef::new(x, n, incx, 0).expect("x view ref")
}

#[inline] 
pub(crate) fn make_view_mut<'a>(x: &'a mut [f32], n: usize, incx: usize) -> VectorMut<'a, f32> { 
    VectorMut::new(x, n, incx, 0).expect("x view ref")
}

#[inline] 
pub(crate) fn make_matview_ref<'a> (
    a: &'a [f32], 
    n_rows: usize,
    n_cols: usize, 
    lda: usize
) ->  MatrixRef<'a, f32> { 
    MatrixRef::new(a, n_rows, n_cols, lda, 0).expect("a view ref")
}

#[inline] 
pub(crate) fn make_matview_mut<'a> (
    a: &'a mut [f32], 
    n_rows: usize,
    n_cols: usize, 
    lda: usize
) ->  MatrixMut<'a, f32> { 
    MatrixMut::new(a, n_rows, n_cols, lda, 0).expect("a view mut")
}

