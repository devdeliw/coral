#[inline] 
fn check_len_f32(n: usize, buf: &mut Vec<f32>) { 
    if buf.len() != n { 
        buf.resize(n, 0.0); 
    }
}

/// Packs a strided vector to a contiguous buffer. 
#[inline] 
pub(crate) fn pack_vector_f32 ( 
    alpha: f32, 
    n: usize, 
    x: &[f32], 
    incx: usize, 
    y: &mut Vec<f32>, 
) {
    if n == 0 { 
        return; 
    } 

    check_len_f32(n, y); 

    let xs_it = x.iter().step_by(incx).take(n); 
    let ys_it = y.iter_mut().take(n); 

    for (xv, yv) in xs_it.zip(ys_it) { 
        *yv = *xv * alpha; 
    }
}
