use super::common::{ 
    make_strided_mat, 
    make_strided_vec, 
    assert_close, 
    CoralResult, 
    ATOL, 
    RTOL
}; 

use blas_src as _; 
use cblas_sys::{cblas_sger, CBLAS_LAYOUT}; 
use coral::types::{VectorRef, MatrixMut}; 
use coral::level2::sger;

#[test] 
fn unit_stride() -> CoralResult { 
    let m = 1024;
    let n = 512; 
    let incx = 1; 
    let incy = 1; 
    let lda = m; 

    let xoff = 2; 
    let yoff = 3; 

    let alpha = 3.1415926535; 

    let xbuf = make_strided_vec(m + xoff, incx); 
    let ybuf = make_strided_vec(n + yoff, incy); 

    let abuf = make_strided_mat(m, n, lda); 

    let mut abuf_coral = abuf.clone(); 
    let mut abuf_cblas = abuf.clone(); 

    let xcoral = VectorRef::new(&xbuf, m, incx, xoff)?; 
    let ycoral = VectorRef::new(&ybuf, n, incy, yoff)?; 
    let acoral = MatrixMut::new(&mut abuf_coral, m, n, lda, 0)?; 

    sger(alpha, acoral, xcoral, ycoral); 

    unsafe { 
        cblas_sger ( 
            CBLAS_LAYOUT::CblasColMajor, 
            m as i32, 
            n as i32,
            alpha, 
            xbuf[xoff..].as_ptr(),
            incx as i32, 
            ybuf[yoff..].as_ptr(), 
            incy as i32, 
            abuf_cblas.as_mut_ptr(), 
            lda as i32
        )
    }

    assert_close(&abuf_coral, &abuf_cblas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn strided() -> CoralResult { 
    let m = 1024;
    let n = 512; 
    let incx = 2; 
    let incy = 3; 
    let lda = m + 32; 

    let alpha = 3.1415926535; 

    let xbuf = make_strided_vec(m, incx); 
    let ybuf = make_strided_vec(n, incy); 

    let abuf = make_strided_mat(m, n, lda); 

    let mut abuf_coral = abuf.clone(); 
    let mut abuf_cblas = abuf.clone(); 

    let xcoral = VectorRef::new(&xbuf, m, incx, 0)?; 
    let ycoral = VectorRef::new(&ybuf, n, incy, 0)?; 
    let acoral = MatrixMut::new(&mut abuf_coral, m, n, lda, 0)?; 

    sger(alpha, acoral, xcoral, ycoral); 

    unsafe { 
        cblas_sger ( 
            CBLAS_LAYOUT::CblasColMajor, 
            m as i32, 
            n as i32,
            alpha, 
            xbuf.as_ptr(),
            incx as i32, 
            ybuf.as_ptr(), 
            incy as i32, 
            abuf_cblas.as_mut_ptr(), 
            lda as i32
        )
    }

    assert_close(&abuf_coral, &abuf_cblas, RTOL, ATOL); 
    Ok(())
}

