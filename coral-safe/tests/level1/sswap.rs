use super::common::{
    make_strided_vec, 
    assert_close, 
    CoralResult,
    RTOL, 
    ATOL, 
}; 

use blas_src as _; 
use cblas_sys::cblas_sswap; 
use coral::level1::sswap; 
use coral::types::VectorMut; 

#[test]
fn unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1;
    let incy = 1; 

    let mut xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let xorig = xbuf.clone(); 
    let yorig = ybuf.clone(); 

    let mut xcblas = xbuf.clone(); 
    let mut ycblas = ybuf.clone(); 

    let xvec = VectorMut::new(&mut xbuf, n, incx, 0)?; 
    let yvec = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    sswap(xvec, yvec);
    unsafe { 
        cblas_sswap ( 
            n as i32, 
            xcblas.as_mut_ptr(), 
            incx as i32, 
            ycblas.as_mut_ptr(), 
            incy as i32
        )
    }

    assert_close(&xbuf, &xcblas, RTOL, ATOL); 
    assert_close(&ybuf, &ycblas, RTOL, ATOL); 
    assert_close(&xbuf, &yorig,  RTOL, ATOL);
    assert_close(&ybuf, &xorig,  RTOL, ATOL); 
    Ok(())
}

#[test]
fn strided() -> CoralResult { 
    let n = 1024; 
    let incx = 3;
    let incy = 5; 

    let mut xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let mut xcblas = xbuf.clone(); 
    let mut ycblas = ybuf.clone(); 

    let xvec = VectorMut::new(&mut xbuf, n, incx, 0)?; 
    let yvec = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    sswap(xvec, yvec);
    unsafe { 
        cblas_sswap ( 
            n as i32, 
            xcblas.as_mut_ptr(), 
            incx as i32, 
            ycblas.as_mut_ptr(), 
            incy as i32
        )
    }

    assert_close(&xbuf, &xcblas, RTOL, ATOL); 
    assert_close(&ybuf, &ycblas, RTOL, ATOL); 
    Ok(())
}

#[test]
fn n_zero() -> CoralResult { 
    let n = 0; 
    let incx = 1;
    let incy = 1; 

    let mut xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let xorig = xbuf.clone(); 
    let yorig = ybuf.clone(); 

    let mut xcblas = xbuf.clone(); 
    let mut ycblas = ybuf.clone(); 

    let xvec = VectorMut::new(&mut xbuf, n, incx, 0)?; 
    let yvec = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    sswap(xvec, yvec);
    unsafe { 
        cblas_sswap ( 
            n as i32, 
            xcblas.as_mut_ptr(), 
            incx as i32, 
            ycblas.as_mut_ptr(), 
            incy as i32
        )
    }

    assert_close(&xbuf, &xcblas, RTOL, ATOL); 
    assert_close(&ybuf, &ycblas, RTOL, ATOL); 
    assert_close(&xbuf, &yorig,  RTOL, ATOL);
    assert_close(&ybuf, &xorig,  RTOL, ATOL); 
    Ok(())
}



