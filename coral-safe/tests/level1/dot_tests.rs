use blas_src as _;
use cblas_sys::cblas_sdot;

use coral_safe::level1::sdot;
use coral_safe::types::VectorRef;
use coral_safe::errors::BufferError;

fn make_strided_vec_f32(
    len: usize,
    inc: usize,
    f: impl Fn(usize) -> f32,
) -> Vec<f32> {
    let mut v   = vec![0.0; (len.saturating_sub(1)) * inc + (len > 0) as usize];
    let mut idx = 0;

    for k in 0..len {
        v[idx] = f(k);
        idx += inc;
    }

    v
}

fn assert_close_f32(
    x: f32,
    y: f32,
    rtol: f32,
    atol: f32,
) {
    let diff = (x - y).abs();
    let tol  = atol + rtol * x.abs().max(y.abs());

    assert!(
        diff <= tol,
        "mismatch: {x} vs {y} (diff={diff}, tol={tol})"
    );
}

const RTOL_F32: f32 = 1e-6;
const ATOL_F32: f32 = 1e-5;

#[test]
fn sdot_unit_stride() -> Result<(), BufferError> {
    let n     = 1024;
    let xbuf  = make_strided_vec_f32(n, 1, |k| 0.05 + 0.03 * k as f32);
    let ybuf  = make_strided_vec_f32(n, 1, |k| 0.10 + 0.06 * k as f32);

    let xview = VectorRef::new(&xbuf, n, 1, 0)?;
    let yview = VectorRef::new(&ybuf, n, 1, 0)?;

    let res_safe = sdot(xview, yview);

    let res_blas = unsafe {
        cblas_sdot(
            n as i32,
            xbuf.as_ptr(),
            1,
            ybuf.as_ptr(),
            1,
        )
    };

    assert_close_f32(res_safe, res_blas, RTOL_F32, ATOL_F32);
    Ok(())
}

#[test]
fn sdot_strided() -> Result<(), BufferError> {
    let n    = 777;
    let incx = 3;
    let incy = 2;

    let xbuf = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * k as f32);
    let ybuf = make_strided_vec_f32(n, incy, |k| -0.3 + 0.002 * k as f32);

    let xview = VectorRef::new(&xbuf, n, incx, 0)?;
    let yview = VectorRef::new(&ybuf, n, incy, 0)?;

    let res_safe = sdot(xview, yview);

    let res_blas = unsafe {
        cblas_sdot(
            n as i32,
            xbuf.as_ptr(),
            incx as i32,
            ybuf.as_ptr(),
            incy as i32,
        )
    };

    assert_close_f32(res_safe, res_blas, RTOL_F32, ATOL_F32);
    Ok(())
}

#[test]
fn sdot_n_zero_returns_zero() -> Result<(), BufferError> {
    let n = 0;

    let xbuf = vec![2.0f32; 4];
    let ybuf = vec![1.0f32; 4];

    let xview = VectorRef::new(&xbuf, n, 1, 0)?;
    let yview = VectorRef::new(&ybuf, n, 1, 0)?;

    let res_safe = sdot(xview, yview);

    let res_blas = unsafe {
        cblas_sdot(
            n as i32,
            xbuf.as_ptr(),
            1,
            ybuf.as_ptr(),
            1,
        )
    };

    assert_close_f32(res_safe, res_blas, RTOL_F32, ATOL_F32);
    Ok(())
}
