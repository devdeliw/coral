use blas_src as _;

use cblas_sys::cblas_scopy;

use coral_safe::level1::scopy;
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

#[test]
fn scopy_unit_stride() -> Result<(), BufferError> {
    let n = 2048;

    let xbuf = make_strided_vec_f32(n, 1, |k| -0.25 + 0.001 * k as f32);

    let mut ybuf_coral = vec![1.0f32; n];
    let mut ybuf_blas  = vec![1.0f32; n];

    let xview = VectorRef::new(&xbuf, n, 1, 0)?;
    let yview = VectorMut::new(&mut ybuf_coral, n, 1, 0)?;

    scopy(xview, yview);

    unsafe {
        cblas_scopy(
            n as i32,
            xbuf.as_ptr(),
            1,
            ybuf_blas.as_mut_ptr(),
            1,
        );
    }

    assert_eq!(ybuf_coral.len(), ybuf_blas.len());
    for (yc, yb) in ybuf_coral.iter().zip(ybuf_blas.iter()) {
        assert!(
            yc.to_bits() == yb.to_bits(),
            "mismatch: coral={yc}, blas={yb}"
        );
    }

    Ok(())
}

#[test]
fn scopy_strided() -> Result<(), BufferError> {
    let n    = 777;
    let incx = 3;
    let incy = 5;

    let xbuf = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * k as f32);

    let mut ybuf_coral = make_strided_vec_f32(n, incy, |_| 1.5);
    let mut ybuf_blas  = ybuf_coral.clone();

    let xview = VectorRef::new(&xbuf, n, incx, 0)?;
    let yview = VectorMut::new(&mut ybuf_coral, n, incy, 0)?;

    scopy(xview, yview);

    unsafe {
        cblas_scopy(
            n as i32,
            xbuf.as_ptr(),
            incx as i32,
            ybuf_blas.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_eq!(ybuf_coral.len(), ybuf_blas.len());
    for (yc, yb) in ybuf_coral.iter().zip(ybuf_blas.iter()) {
        assert!(
            yc.to_bits() == yb.to_bits(),
            "mismatch: coral={yc}, blas={yb}"
        );
    }

    Ok(())
}

#[test]
fn scopy_n_zero_noop() -> Result<(), BufferError> {
    let n    = 0;
    let incx = 3;
    let incy = 2;

    let xbuf = make_strided_vec_f32(4, incx, |k| -0.1 * k as f32);

    let mut ybuf_coral = make_strided_vec_f32(4, incy, |k| 2.0 + 0.01 * k as f32);
    let mut ybuf_blas  = ybuf_coral.clone();

    let xview = VectorRef::new(&xbuf, n, incx, 0)?;
    let yview = VectorMut::new(&mut ybuf_coral, n, incy, 0)?;

    scopy(xview, yview);

    unsafe {
        cblas_scopy(
            n as i32,
            xbuf.as_ptr(),
            incx as i32,
            ybuf_blas.as_mut_ptr(),
            incy as i32,
        );
    }

    assert_eq!(ybuf_coral.len(), ybuf_blas.len());
    for (yc, yb) in ybuf_coral.iter().zip(ybuf_blas.iter()) {
        assert!(
            yc.to_bits() == yb.to_bits(),
            "mismatch after n=0 call: coral={yc}, blas={yb}"
        );
    }

    Ok(())
}

