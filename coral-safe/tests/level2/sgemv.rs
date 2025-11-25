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
use coral::types::{VectorMut, VectorRef, MatrixRef, CoralTranspose}; 
use coral::level2::sgemv; 

#[test]
fn unit_stride_n() -> CoralResult { 
    let m = 1024;
    let n = 512; 
    let incx = 1;
    let incy = 1; 
    let lda = m; 

    let alpha = 3.14159265;
    let beta  = 2.71828182; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_strided_mat(m, n, lda); 

    let mut ybuf = make_strided_vec(m, incy);
    let mut ycblas = ybuf.clone(); 

    let xcoral = VectorRef::new(&xbuf, n, incx, 0)?; 
    let acoral = MatrixRef::new(&abuf, m, n, lda, 0)?;
    let ycoral = VectorMut::new(&mut ybuf, m, incy, 0)?; 

    sgemv(CoralTranspose::NoTrans, alpha, beta, acoral, xcoral, ycoral); 

    unsafe { 
        cblas_sgemv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_TRANSPOSE::CblasNoTrans, 
            m as i32, 
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
fn strided_n() -> CoralResult { 
    let m = 1024;
    let n = 512;

    let incx = 2;
    let incy = 3; 
    let lda = m + 32; 

    let alpha = 3.14159265;
    let beta  = 2.71828182; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_strided_mat(m, n, lda); 

    let mut ybuf = make_strided_vec(m, incy);
    let mut ycblas = ybuf.clone(); 

    let xcoral = VectorRef::new(&xbuf, n, incx, 0)?; 
    let acoral = MatrixRef::new(&abuf, m, n, lda, 0)?;
    let ycoral = VectorMut::new(&mut ybuf, m, incy, 0)?; 

    sgemv(CoralTranspose::NoTrans, alpha, beta, acoral, xcoral, ycoral); 
    unsafe { 
        cblas_sgemv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_TRANSPOSE::CblasNoTrans, 
            m as i32, 
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

#[test]
fn unit_stride_t() -> CoralResult { 
    let m = 1024;
    let n = 512; 
    let incx = 1;
    let incy = 1; 
    let lda = m; 

    let alpha = 3.14159265;
    let beta  = 2.71828182; 

    let xbuf = make_strided_vec(m, incx); 
    let abuf = make_strided_mat(m, n, lda); 

    let mut ybuf = make_strided_vec(n, incy);
    let mut ycblas = ybuf.clone(); 

    let xcoral = VectorRef::new(&xbuf, m, incx, 0)?; 
    let acoral = MatrixRef::new(&abuf, m, n, lda, 0)?;
    let ycoral = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    sgemv(CoralTranspose::Trans, alpha, beta, acoral, xcoral, ycoral); 

    unsafe { 
        cblas_sgemv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_TRANSPOSE::CblasTrans, 
            m as i32, 
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
fn strided_t() -> CoralResult { 
    let m = 1024; 
    let n = 512;
    let incx = 2;
    let incy = 3; 
    let lda = m + 32; 

    let alpha = 3.14159265;
    let beta  = 2.71828182; 

    let xbuf = make_strided_vec(m, incx); 
    let abuf = make_strided_mat(m, n, lda); 

    let mut ybuf = make_strided_vec(n, incy);
    let mut ycblas = ybuf.clone(); 

    let xcoral = VectorRef::new(&xbuf, m, incx, 0)?; 
    let acoral = MatrixRef::new(&abuf, m, n, lda, 0)?;
    let ycoral = VectorMut::new(&mut ybuf, n, incy, 0)?; 

    sgemv(CoralTranspose::Trans, alpha, beta, acoral, xcoral, ycoral); 
    unsafe { 
        cblas_sgemv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_TRANSPOSE::CblasTrans, 
            m as i32, 
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
