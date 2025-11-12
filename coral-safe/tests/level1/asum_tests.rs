use blas_src as _;
use cblas_sys::cblas_sasum;

use coral_safe::level1::sasum;
use coral_safe::types::VectorRef;
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
fn sasum_unit_stride() -> Result<(), BufferError> {
    let n = 1024;

    // include negatives test abs
    let xbuf = make_strided_vec_f32(
        n,
        1,
        |k| if k % 2 == 0 { 
            0.05 + 0.03 * k as f32 
        } else {
            -(0.05 + 0.03 * k as f32)
        },
    );

    let xview = VectorRef::new(&xbuf, n, 1, 0)?;

    let coral = sasum(xview);

    let blas = unsafe {
        cblas_sasum(
            n as i32,
            xbuf.as_ptr(),
            1,
        )
    };

    assert_allclose_f32(&[coral], &[blas], RTOL_F32, ATOL_F32);
    Ok(())
}

#[test]
fn sasum_strided() -> Result<(), BufferError> {
    let n    = 777;
    let incx = 3;

    // alternating signs
    let xbuf = make_strided_vec_f32(
        n,
        incx,
        |k| 0.2 - 0.001 * k as f32 * if k % 2 == 0 { 
            1.0
        } else { 
            -1.0 
        },
    );

    let xview = VectorRef::new(&xbuf, n, incx, 0)?;

    let coral = sasum(xview);

    let blas = unsafe {
        cblas_sasum(
            n as i32,
            xbuf.as_ptr(),
            incx as i32,
        )
    };

    assert_allclose_f32(&[coral], &[blas], RTOL_F32, ATOL_F32);
    Ok(())
}

#[test]
fn sasum_n_zero_noop() -> Result<(), BufferError> {
    let n = 0;

    let xbuf = vec![2.0; 4];
    let xview = VectorRef::new(&xbuf, n, 1, 0)?;

    let coral = sasum(xview);

    let blas = unsafe {
        cblas_sasum(
            n as i32,
            xbuf.as_ptr(),
            1,
        )
    };

    assert_allclose_f32(&[coral], &[blas], RTOL_F32, ATOL_F32);
    Ok(())
}

