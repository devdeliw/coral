use rand::thread_rng;
use rand::distributions::{Distribution, Standard, Uniform}; 
use coral::types::{VectorMut, VectorRef, MatrixMut, MatrixRef};
use coral::types::{CoralTriangular, CoralDiagonal}; 

/// for sweeps
#[allow(dead_code)]
pub const SIZES: &[usize] = &[
    128, 256, 384, 512, 640, 768, 896, 1024,
    1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
];

/// Make a `Vec<f32>` buffer of length `len` 
/// with stride `inc` with randomized elems
#[inline]
#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) fn make_strided_mat (
    n_rows: usize,
    n_cols: usize, 
    lda: usize, 
) -> Vec<f32> {

    debug_assert!(n_rows <= lda, "# rows must be <= lda"); 

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

#[allow(dead_code)]
pub fn make_triangular_mat(
    uplo: CoralTriangular,
    diag: CoralDiagonal,
    n: usize,
    lda: usize,
) -> Vec<f32> {
    if n == 0 || lda == 0 {
        return vec![1.0; 1];
    }

    let unit = diag.is_unit();
    let mut buf = vec![0.0; lda * n];

    let mut rng = thread_rng();

    let diag_dist = Uniform::new(1.0, 2.0);

    // keep well conditioned
    let eps = 1e-3 / (n.max(1) as f32);
    let off_dist = Uniform::new(-eps, eps);

    for j in 0..n {
        match uplo {
            CoralTriangular::Upper => {
                for i in 0..=j {
                    let idx = i + j * lda;
                    if i == j {
                        buf[idx] = if unit {
                            1.0
                        } else {
                            diag_dist.sample(&mut rng)
                        };
                    } else {
                        buf[idx] = off_dist.sample(&mut rng);
                    }
                }
            }
            CoralTriangular::Lower => {
                for i in j..n {
                    let idx = i + j * lda;
                    if i == j {
                        buf[idx] = if unit {
                            1.0
                        } else {
                            diag_dist.sample(&mut rng)
                        };
                    } else {
                        buf[idx] = off_dist.sample(&mut rng);
                    }
                }
            }
        }
    }

    buf
}


#[inline]
#[allow(dead_code)]
pub(crate) fn bytes(n: usize, alpha: usize) -> u64 { 
    (alpha * n * std::mem::size_of::<f32>()) as u64
}

#[inline]
#[allow(dead_code)]
pub(crate) fn bytes_decimal(n: f32, alpha: f32) -> u64 { 
    (alpha * n * (std::mem::size_of::<f32>() as f32)) as u64
}

#[inline] 
#[allow(dead_code)]
pub(crate) fn make_view_ref<'a>(x: &'a [f32], n: usize, incx: usize) -> VectorRef<'a, f32> { 
    VectorRef::new(x, n, incx, 0).expect("x view ref")
}

#[inline] 
#[allow(dead_code)]
pub(crate) fn make_view_mut<'a>(x: &'a mut [f32], n: usize, incx: usize) -> VectorMut<'a, f32> { 
    VectorMut::new(x, n, incx, 0).expect("x view ref")
}

#[inline] 
#[allow(dead_code)]
pub(crate) fn make_matview_ref<'a> (
    a: &'a [f32], 
    n_rows: usize,
    n_cols: usize, 
    lda: usize
) ->  MatrixRef<'a, f32> { 
    MatrixRef::new(a, n_rows, n_cols, lda, 0).expect("a view ref")
}

#[inline] 
#[allow(dead_code)]
pub(crate) fn make_matview_mut<'a> (
    a: &'a mut [f32], 
    n_rows: usize,
    n_cols: usize, 
    lda: usize
) ->  MatrixMut<'a, f32> { 
    MatrixMut::new(a, n_rows, n_cols, lda, 0).expect("a view mut")
}

