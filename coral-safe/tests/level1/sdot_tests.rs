use super::common::{
    make_strided_vec, 
    assert_close, 
    CoralResult,
    RTOL, 
    ATOL, 
}; 

use blas_src as _; 
use cblas_sys::cblas_sdot; 
use coral_safe::level1::sdot; 
use coral_safe::types::VectorRef; 

#[test]
fn unit_stride() -> CoralResult { 
    let n = 1024;
    let incx = 1; 
    let incy = 1; 
    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 
    let xvec = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yvec = VectorRef::new(&ybuf, n, incy, 0)?; 

    let coral_val = sdot(xvec, yvec); 
    let cblas_val = unsafe { 
        cblas_sdot ( 
            n as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
            ybuf.as_ptr(), 
            incy as i32, 
        )
    }; 

    assert_close(&[coral_val], &[cblas_val], RTOL, ATOL); 
    Ok(())
}

#[test]
fn strided() -> CoralResult { 
    let n = 1024; 
    let incx = 3; 
    let incy = 5; 
    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 
    let xvec = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yvec = VectorRef::new(&ybuf, n, incy, 0)?; 

    let coral_val = sdot(xvec, yvec); 
    let cblas_val = unsafe { 
        cblas_sdot ( 
            n as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
            ybuf.as_ptr(), 
            incy as i32, 
        )
    }; 

    assert_close(&[coral_val], &[cblas_val], RTOL, ATOL); 
    Ok(())
}

#[test]
fn n_zero() -> CoralResult { 
    let n = 0; 
    let incx = 1; 
    let incy = 1; 
    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 
    let xvec = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yvec = VectorRef::new(&ybuf, n, incy, 0)?; 

    let coral_val = sdot(xvec, yvec); 
    let cblas_val = unsafe { 
        cblas_sdot ( 
            n as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
            ybuf.as_ptr(), 
            incy as i32, 
        )
    }; 

    assert_close(&[coral_val], &[cblas_val], RTOL, ATOL); 
    Ok(())
}




