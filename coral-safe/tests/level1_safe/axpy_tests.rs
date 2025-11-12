use blas_src as _; 
use cblas_sys::cblas_saxpy;

use coral_safe::level1::saxpy;               
use coral_safe::types::{VectorRef, VectorMut};
use coral_safe::errors::BufferError;

fn make_strided_vec_f32(
    len : usize,
    inc : usize, 
    f   : impl Fn(usize) -> f32
) -> Vec<f32> {
    let mut v   = vec![0.0; (len.saturating_sub(1)) * inc + (len > 0) as usize];
    let mut idx = 0;

    for k in 0..len {
        v[idx] = f(k);
        idx += inc;
    }

    v
}

fn assert_allclose_f32(
    a: &[f32],
    b: &[f32], 
    rtol: f32, 
    atol: f32
) {
    assert_eq!(a.len(), b.len());

    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol  = atol + rtol * x.abs().max(y.abs());

        assert!(
            diff <= tol, 
            "mismatch at {i}: {x} vs {y} (diff={diff}, tol={tol})"
        );
    }
}


const RTOL_F32: f32 = 1e-6;
const ATOL_F32: f32 = 1e-5;


#[test]
fn saxpy_unit_stride() -> Result<(), BufferError> {
    let n = 1024;
    let alpha = 3.1415926;

    let xbuf = make_strided_vec_f32(n, 1, |k| 0.05 + 0.03 * k as f32);
    let ybuf = make_strided_vec_f32(n, 1, |k| 0.10 + 0.06 * k as f32);

    let mut yc = ybuf.clone(); 
    let mut yr = ybuf.clone(); 

    let xview = VectorRef::new(&xbuf, n, 1, 0)?;
    let yview = VectorMut::new(&mut yc, n, 1, 0)?;

    saxpy(alpha, xview, yview);

    unsafe {
        cblas_saxpy(
            n as i32,
            alpha,
            xbuf.as_ptr(),
            1,
            yr.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    Ok(())
}

#[test]
fn saxpy_strided() -> Result<(), BufferError> {
    let n = 777;
    let alpha = -0.75;
    let incx  = 3;
    let incy  = 2;

    let xbuf = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * k as f32);
    let ybuf = make_strided_vec_f32(n, incy, |k| -0.3 + 0.002 * k as f32);

    let mut yc = ybuf.clone();
    let mut yr = ybuf.clone();

    let xview = VectorRef::new(&xbuf, n, incx, 0)?;
    let yview = VectorMut::new(&mut yc, n, incy, 0)?;

    saxpy(alpha, xview, yview);

    unsafe {
        cblas_saxpy(
            n as i32,
            alpha,
            xbuf.as_ptr(),
            incx as i32,
            yr.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    Ok(())
}

#[test]
fn saxpy_n_zero_noop() -> Result<(), BufferError> {
    let n = 0;
    let alpha: f32 = 1.23;

    let xbuf   = vec![2.0; 4];
    let mut yc = vec![1.0; 4];
    let mut yr = vec![1.0; 4];

    let xview = VectorRef::new(&xbuf, n, 1, 0)?;
    let yview = VectorMut::new(&mut yc, n, 1, 0)?;

    saxpy(alpha, xview, yview);

    unsafe {
        cblas_saxpy(
            n as i32,
            alpha,
            xbuf.as_ptr(),
            1,
            yr.as_mut_ptr(),
            1,
        );
    }

    assert_allclose_f32(&yc, &yr, RTOL_F32, ATOL_F32);
    Ok(())
}

