use blas_src as _;
use cblas_sys::cblas_isamax;

use coral_safe::level1::isamax;
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

#[test]
fn isamax_unit_stride() -> Result<(), BufferError> {
    let n = 2048;

    let xbuf = make_strided_vec_f32(n, 1, |k| -0.25 + 0.001 * k as f32);

    let xview = VectorRef::new(&xbuf, n, 1, 0)?;
    let rc = isamax(xview);

    let rb = unsafe {
        cblas_isamax(
            n as i32,
            xbuf.as_ptr(),
            1,
        ) as usize
    };

    assert_eq!(rc, rb);
    Ok(())
}

#[test]
fn isamax_strided() -> Result<(), BufferError> {
    let n    = 777;
    let incx = 3;

    let xbuf = make_strided_vec_f32(n, incx, |k| 0.2 - 0.001 * k as f32);

    let xview = VectorRef::new(&xbuf, n, incx, 0)?;
    let rc = isamax(xview);

    let rb = unsafe {
        cblas_isamax(
            n as i32,
            xbuf.as_ptr(),
            incx as i32,
        ) as usize
    };

    assert_eq!(rc, rb);
    Ok(())
}

#[test]
fn isamax_n_zero_noop() -> Result<(), BufferError> {
    let n = 0;

    let xbuf = vec![2.0; 4];

    let xview = VectorRef::new(&xbuf, n, 1, 0)?;
    let rc = isamax(xview);

    let rb = unsafe {
        cblas_isamax(
            n as i32,
            xbuf.as_ptr(),
            1,
        ) as usize
    } as usize;

    assert_eq!(rc, rb);
    Ok(())
}
