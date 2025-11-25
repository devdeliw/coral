use super::common::{
    make_strided_mat, 
    make_strided_vec, 
    assert_close, 
    CoralResult, 
    ATOL, 
    RTOL, 
}; 

use blas_src as _; 
use cblas_sys::{cblas_sgemv, CBLAS_LAYOUT, CBLAS_TRANSPOSE}; 
use coral::fused::sdotf; 
use coral::types::{VectorMut, VectorRef, MatrixRef}; 

/// AXPYF equivalent to GEMV with alpha, beta = 1.0 
#[test]
fn unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1;
    let incy = 1; 
    let lda = n; 

    let alpha = 1.0;
    let beta  = 1.0; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_strided_mat(n, n, lda); 

    let mut ybuf = make_strided_vec(n, incy);
    let mut ycblas = ybuf.clone(); 

    let xcoral = VectorRef::new(&xbuf, n, incx, 0)?; 
    let acoral = MatrixRef::new(&abuf, n, n, lda, 0)?;
    let ycoral = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    sdotf(acoral, xcoral, ycoral); 
    unsafe { 
        cblas_sgemv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_TRANSPOSE::CblasTrans, 
            n as i32, 
            n as i32, 
            alpha, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
            beta, 
            ycblas.as_mut_ptr(), 
            incy as i32, 
        );
    };

    assert_close(&ybuf, &ycblas, RTOL, ATOL);

    Ok(())
}  

#[test]
fn strided() -> CoralResult { 
    let n = 1024; 
    let incx = 2;
    let incy = 3; 
    let lda = n; 

    let alpha = 1.0;
    let beta  = 1.0; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_strided_mat(n, n, lda); 

    let mut ybuf = make_strided_vec(n, incy);
    let mut ycblas = ybuf.clone(); 

    let xcoral = VectorRef::new(&xbuf, n, incx, 0)?; 
    let acoral = MatrixRef::new(&abuf, n, n, lda, 0)?;
    let ycoral = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    sdotf(acoral, xcoral, ycoral); 
    unsafe { 
        cblas_sgemv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_TRANSPOSE::CblasTrans, 
            n as i32, 
            n as i32, 
            alpha, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
            beta, 
            ycblas.as_mut_ptr(), 
            incy as i32, 
        )
    };

    assert_close(&ybuf, &ycblas, RTOL, ATOL);

    Ok(())
}   




