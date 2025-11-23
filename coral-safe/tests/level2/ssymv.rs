use super::common::{ 
    make_triangular_mat, 
    make_strided_vec, 
    assert_close, 
    CoralResult,
    RTOL, 
    ATOL, 
};

use blas_src as _; 
use cblas_sys::{cblas_ssymv, CBLAS_UPLO, CBLAS_LAYOUT};
use coral_safe::types::{MatrixRef, VectorRef, VectorMut, CoralDiagonal, CoralTriangular};
use coral_safe::level2::ssymv; 


#[test] 
fn upper_unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let incy = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Upper; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 
    let abuf = make_triangular_mat(uplo, CoralDiagonal::NonUnit, n, lda); 

    let mut ysafe = ybuf.clone(); 
    let mut yblas = ybuf.clone(); 

    let acoral = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xcoral = VectorRef::new(&xbuf, n, incx, 0)?; 
    let ycoral = VectorMut::new(&mut ysafe, n, incy, 0)?; 

    let alpha = 3.1415; 
    let beta = 2.71828;

    ssymv(uplo, alpha, beta, acoral, xcoral, ycoral); 

    unsafe { 
        cblas_ssymv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            n as i32, 
            alpha, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
            beta, 
            yblas.as_mut_ptr(), 
            incy as i32,
        );
    }

    assert_close(&ysafe, &yblas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn lower_unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let incy = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Lower; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 
    let abuf = make_triangular_mat(uplo, CoralDiagonal::NonUnit, n, lda); 

    let mut ysafe = ybuf.clone(); 
    let mut yblas = ybuf.clone(); 

    let acoral = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xcoral = VectorRef::new(&xbuf, n, incx, 0)?; 
    let ycoral = VectorMut::new(&mut ysafe, n, incy, 0)?; 

    let alpha = 3.1415; 
    let beta = 2.71828;

    ssymv(uplo, alpha, beta, acoral, xcoral, ycoral); 

    unsafe { 
        cblas_ssymv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            n as i32, 
            alpha, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
            beta, 
            yblas.as_mut_ptr(), 
            incy as i32,
        );
    }

    assert_close(&ysafe, &yblas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn upper_strided() -> CoralResult { 
    let n = 1024; 
    let incx = 2; 
    let incy = 3; 
    let lda = n; 

    let uplo = CoralTriangular::Upper; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 
    let abuf = make_triangular_mat(uplo, CoralDiagonal::NonUnit, n, lda); 

    let mut ysafe = ybuf.clone(); 
    let mut yblas = ybuf.clone(); 

    let acoral = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xcoral = VectorRef::new(&xbuf, n, incx, 0)?; 
    let ycoral = VectorMut::new(&mut ysafe, n, incy, 0)?; 

    let alpha = 3.1415; 
    let beta = 2.71828;

    ssymv(uplo, alpha, beta, acoral, xcoral, ycoral); 

    unsafe { 
        cblas_ssymv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            n as i32, 
            alpha, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
            beta, 
            yblas.as_mut_ptr(), 
            incy as i32,
        );
    }

    assert_close(&ysafe, &yblas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn lower_strided() -> CoralResult { 
    let n = 1024; 
    let incx = 2; 
    let incy = 3; 
    let lda = n + 32; 

    let uplo = CoralTriangular::Lower; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 
    let abuf = make_triangular_mat(uplo, CoralDiagonal::NonUnit, n, lda); 

    let mut ysafe = ybuf.clone(); 
    let mut yblas = ybuf.clone(); 

    let acoral = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xcoral = VectorRef::new(&xbuf, n, incx, 0)?; 
    let ycoral = VectorMut::new(&mut ysafe, n, incy, 0)?; 

    let alpha = 3.1415; 
    let beta = 2.71828;

    ssymv(uplo, alpha, beta, acoral, xcoral, ycoral); 

    unsafe { 
        cblas_ssymv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            n as i32, 
            alpha, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
            beta, 
            yblas.as_mut_ptr(), 
            incy as i32,
        );
    }

    assert_close(&ysafe, &yblas, RTOL, ATOL); 
    Ok(())
}





