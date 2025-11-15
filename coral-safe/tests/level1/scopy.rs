use super::common::{
    make_strided_vec, 
    assert_close, 
    CoralResult,
    RTOL, 
    ATOL, 
}; 

use blas_src as _; 
use cblas_sys::cblas_scopy; 
use coral_safe::level1::scopy; 
use coral_safe::types::{VectorRef, VectorMut}; 

#[test]
fn unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let incy = 1; 
    
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let xcblas = xbuf.clone(); 
    let mut ycblas = ybuf.clone(); 

    let xvec = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yvec = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    scopy(xvec, yvec); 
    unsafe { 
        cblas_scopy ( 
            n as i32, 
            xcblas.as_ptr(), 
            incx as i32, 
            ycblas.as_mut_ptr(), 
            incy as i32
        )
    };

    assert_close(&ybuf, &ycblas, RTOL, ATOL); 
    Ok(())
}

#[test]
fn strided() -> CoralResult { 
    let n = 1024; 
    let incx = 3; 
    let incy = 5; 
    
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let xcblas = xbuf.clone(); 
    let mut ycblas = ybuf.clone(); 

    let xvec = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yvec = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    scopy(xvec, yvec); 
    unsafe { 
        cblas_scopy ( 
            n as i32, 
            xcblas.as_ptr(), 
            incx as i32, 
            ycblas.as_mut_ptr(), 
            incy as i32
        )
    };

    assert_close(&ybuf, &ycblas, RTOL, ATOL); 
    Ok(())
}

#[test]
fn n_zero() -> CoralResult { 
    let n = 0; 
    let incx = 1; 
    let incy = 1; 
    
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let xcblas = xbuf.clone(); 
    let mut ycblas = ybuf.clone(); 

    let xvec = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yvec = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    scopy(xvec, yvec); 
    unsafe { 
        cblas_scopy ( 
            n as i32, 
            xcblas.as_ptr(), 
            incx as i32, 
            ycblas.as_mut_ptr(), 
            incy as i32
        )
    };

    assert_close(&ybuf, &ycblas, RTOL, ATOL); 
    Ok(())
}

