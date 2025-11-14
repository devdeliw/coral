use blas_src as _;

use cblas_sys::{
    cblas_srot,
    cblas_srotg,
    cblas_srotm,
    cblas_srotmg,
};

use coral_safe::level1::{
    srot,
    srotg,
    srotm,
    srotmg,
};

use coral_safe::types::VectorMut;
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
fn srot_unit_stride() -> Result<(), BufferError> {
    let n    = 1024;
    let c    = 0.8f32;
    let s    = 0.6f32;

    let xbuf  = make_strided_vec_f32(n, 1, |k| 0.05 + 0.03 * k as f32);
    let ybuf  = make_strided_vec_f32(n, 1, |k| 0.10 + 0.06 * k as f32);

    let mut xbuf_safe = xbuf.clone();
    let mut ybuf_safe = ybuf.clone();
    let mut xbuf_blas = xbuf.clone();
    let mut ybuf_blas = ybuf.clone();

    {
        let xview = VectorMut::new(&mut xbuf_safe, n, 1, 0)?;
        let yview = VectorMut::new(&mut ybuf_safe, n, 1, 0)?;
        srot(xview, yview, c, s);
    }

    unsafe {
        cblas_srot(
            n as i32,
            xbuf_blas.as_mut_ptr(),
            1,
            ybuf_blas.as_mut_ptr(),
            1,
            c,
            s,
        );
    }

    for i in 0..xbuf_safe.len() {
        assert_close_f32(xbuf_safe[i], xbuf_blas[i], RTOL_F32, ATOL_F32);
    }
    for i in 0..ybuf_safe.len() {
        assert_close_f32(ybuf_safe[i], ybuf_blas[i], RTOL_F32, ATOL_F32);
    }

    Ok(())
}

#[test]
fn srot_strided() -> Result<(), BufferError> {
    let n    = 777;
    let incx = 3;
    let incy = 2;
    let c    = 0.7f32;
    let s    = -0.4f32;

    let xbuf = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * k as f32);
    let ybuf = make_strided_vec_f32(n, incy, |k| -0.3 + 0.002 * k as f32);

    let mut xbuf_safe = xbuf.clone();
    let mut ybuf_safe = ybuf.clone();
    let mut xbuf_blas = xbuf.clone();
    let mut ybuf_blas = ybuf.clone();

    {
        let xview = VectorMut::new(&mut xbuf_safe, n, incx, 0)?;
        let yview = VectorMut::new(&mut ybuf_safe, n, incy, 0)?;
        srot(xview, yview, c, s);
    }

    unsafe {
        cblas_srot(
            n as i32,
            xbuf_blas.as_mut_ptr(),
            incx as i32,
            ybuf_blas.as_mut_ptr(),
            incy as i32,
            c,
            s,
        );
    }

    for i in 0..xbuf_safe.len() {
        assert_close_f32(xbuf_safe[i], xbuf_blas[i], RTOL_F32, ATOL_F32);
    }
    for i in 0..ybuf_safe.len() {
        assert_close_f32(ybuf_safe[i], ybuf_blas[i], RTOL_F32, ATOL_F32);
    }

    Ok(())
}

#[test]
fn srot_n_zero_noop() -> Result<(), BufferError> {
    let n    = 0;
    let c    = 0.9f32;
    let s    = 0.1f32;

    let xbuf = vec![2.0f32; 4];
    let ybuf = vec![1.0f32; 4];

    let mut xbuf_safe = xbuf.clone();
    let mut ybuf_safe = ybuf.clone();
    let mut xbuf_blas = xbuf.clone();
    let mut ybuf_blas = ybuf.clone();

    {
        let xview = VectorMut::new(&mut xbuf_safe, n, 1, 0)?;
        let yview = VectorMut::new(&mut ybuf_safe, n, 1, 0)?;
        srot(xview, yview, c, s);
    }

    unsafe {
        cblas_srot(
            n as i32,
            xbuf_blas.as_mut_ptr(),
            1,
            ybuf_blas.as_mut_ptr(),
            1,
            c,
            s,
        );
    }

    for i in 0..xbuf_safe.len() {
        assert_close_f32(xbuf_safe[i], xbuf_blas[i], RTOL_F32, ATOL_F32);
    }
    for i in 0..ybuf_safe.len() {
        assert_close_f32(ybuf_safe[i], ybuf_blas[i], RTOL_F32, ATOL_F32);
    }

    Ok(())
}

#[test]
fn srotg_matches_cblas() -> Result<(), BufferError> {
    let mut a_safe = 0.5f32;
    let mut b_safe = -1.25f32;
    let mut c_safe = 0.0f32;
    let mut s_safe = 0.0f32;

    srotg(&mut a_safe, &mut b_safe, &mut c_safe, &mut s_safe);

    let mut a_blas = 0.5f32;
    let mut b_blas = -1.25f32;
    let mut c_blas = 0.0f32;
    let mut s_blas = 0.0f32;

    unsafe {
        cblas_srotg(
            &mut a_blas as *mut f32,
            &mut b_blas as *mut f32,
            &mut c_blas as *mut f32,
            &mut s_blas as *mut f32,
        );
    }

    assert_close_f32(a_safe, a_blas, RTOL_F32, ATOL_F32);
    assert_close_f32(b_safe, b_blas, RTOL_F32, ATOL_F32);
    assert_close_f32(c_safe, c_blas, RTOL_F32, ATOL_F32);
    assert_close_f32(s_safe, s_blas, RTOL_F32, ATOL_F32);

    Ok(())
}

#[test]
fn srotm_unit_stride() -> Result<(), BufferError> {
    let n = 512;

    let param = [
        -1.0f32,  
        0.9f32,   
        -0.3f32,  
        0.4f32,   
        1.1f32,   
    ];

    let xbuf = make_strided_vec_f32(n, 1, |k| 0.01 * k as f32 - 0.5);
    let ybuf = make_strided_vec_f32(n, 1, |k| -0.02 * k as f32 + 0.3);

    let mut xbuf_safe = xbuf.clone();
    let mut ybuf_safe = ybuf.clone();
    let mut xbuf_blas = xbuf.clone();
    let mut ybuf_blas = ybuf.clone();

    {
        let xview = VectorMut::new(&mut xbuf_safe, n, 1, 0)?;
        let yview = VectorMut::new(&mut ybuf_safe, n, 1, 0)?;
        srotm(xview, yview, &param);
    }

    unsafe {
        cblas_srotm(
            n as i32,
            xbuf_blas.as_mut_ptr(),
            1,
            ybuf_blas.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    for i in 0..xbuf_safe.len() {
        assert_close_f32(xbuf_safe[i], xbuf_blas[i], RTOL_F32, ATOL_F32);
    }
    for i in 0..ybuf_safe.len() {
        assert_close_f32(ybuf_safe[i], ybuf_blas[i], RTOL_F32, ATOL_F32);
    }

    Ok(())
}

#[test]
fn srotm_strided() -> Result<(), BufferError> {
    let n    = 321;
    let incx = 2;
    let incy = 3;

    let param = [
        0.0f32,   
        0.0f32,   
        -0.2f32,  
        0.5f32,   
        0.0f32,   
    ];

    let xbuf = make_strided_vec_f32(n, incx, |k| 0.15 - 0.0005 * k as f32);
    let ybuf = make_strided_vec_f32(n, incy, |k| -0.05 + 0.0007 * k as f32);

    let mut xbuf_safe = xbuf.clone();
    let mut ybuf_safe = ybuf.clone();
    let mut xbuf_blas = xbuf.clone();
    let mut ybuf_blas = ybuf.clone();

    {
        let xview = VectorMut::new(&mut xbuf_safe, n, incx, 0)?;
        let yview = VectorMut::new(&mut ybuf_safe, n, incy, 0)?;
        srotm(xview, yview, &param);
    }

    unsafe {
        cblas_srotm(
            n as i32,
            xbuf_blas.as_mut_ptr(),
            incx as i32,
            ybuf_blas.as_mut_ptr(),
            incy as i32,
            param.as_ptr(),
        );
    }

    for i in 0..xbuf_safe.len() {
        assert_close_f32(xbuf_safe[i], xbuf_blas[i], RTOL_F32, ATOL_F32);
    }
    for i in 0..ybuf_safe.len() {
        assert_close_f32(ybuf_safe[i], ybuf_blas[i], RTOL_F32, ATOL_F32);
    }

    Ok(())
}

#[test]
fn srotm_n_zero_noop() -> Result<(), BufferError> {
    let n    = 0usize;

    let param = [
        -2.0f32,  
        1.0f32,
        0.0f32,
        0.0f32,
        1.0f32,
    ];

    let xbuf = vec![3.0f32; 5];
    let ybuf = vec![-2.0f32; 5];

    let mut xbuf_safe = xbuf.clone();
    let mut ybuf_safe = ybuf.clone();
    let mut xbuf_blas = xbuf.clone();
    let mut ybuf_blas = ybuf.clone();

    {
        let xview = VectorMut::new(&mut xbuf_safe, n, 1, 0)?;
        let yview = VectorMut::new(&mut ybuf_safe, n, 1, 0)?;
        srotm(xview, yview, &param);
    }

    unsafe {
        cblas_srotm(
            n as i32,
            xbuf_blas.as_mut_ptr(),
            1,
            ybuf_blas.as_mut_ptr(),
            1,
            param.as_ptr(),
        );
    }

    for i in 0..xbuf_safe.len() {
        assert_close_f32(xbuf_safe[i], xbuf_blas[i], RTOL_F32, ATOL_F32);
    }
    for i in 0..ybuf_safe.len() {
        assert_close_f32(ybuf_safe[i], ybuf_blas[i], RTOL_F32, ATOL_F32);
    }

    Ok(())
}

#[test]
fn srotmg_matches_cblas() -> Result<(), BufferError> {
    let d1_0 = 1.5f32;
    let d2_0 = 2.0f32;
    let x1_0 = -0.75f32;
    let y1   = 0.5f32;

    // Safe version
    let mut d1_safe    = d1_0;
    let mut d2_safe    = d2_0;
    let mut x1_safe    = x1_0;
    let mut param_safe = [0.0f32; 5];

    srotmg(
        &mut d1_safe,
        &mut d2_safe,
        &mut x1_safe,
        y1,
        &mut param_safe,
    );

    let mut d1_blas    = d1_0;
    let mut d2_blas    = d2_0;
    let mut x1_blas    = x1_0;
    let y1_blas        = y1;
    let mut param_blas = [0.0f32; 5];

    unsafe {
        cblas_srotmg(
            &mut d1_blas as *mut f32,
            &mut d2_blas as *mut f32,
            &mut x1_blas as *mut f32,
            y1_blas,
            param_blas.as_mut_ptr(),
        );
    }

    assert_close_f32(d1_safe, d1_blas, RTOL_F32, ATOL_F32);
    assert_close_f32(d2_safe, d2_blas, RTOL_F32, ATOL_F32);
    assert_close_f32(x1_safe, x1_blas, RTOL_F32, ATOL_F32);

    for i in 0..5 {
        assert_close_f32(param_safe[i], param_blas[i], RTOL_F32, ATOL_F32);
    }

    Ok(())
}
