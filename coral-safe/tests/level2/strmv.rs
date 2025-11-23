use super::common::{
    make_triangular_mat, 
    make_strided_vec, 
    assert_close, 
    CoralResult, 
    ATOL, 
    RTOL
};

use blas_src as _; 
use cblas_sys::{cblas_strmv, CBLAS_DIAG, CBLAS_TRANSPOSE, CBLAS_UPLO, CBLAS_LAYOUT}; 
use coral_safe::level2::strmv; 
use coral_safe::types::{CoralDiagonal, CoralTranspose, CoralTriangular, VectorMut, MatrixRef}; 


#[test] 
fn upper_nonunit_n() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Upper; 
    let diag = CoralDiagonal::NonUnit;
    let trans = CoralTranspose::NoTrans; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(uplo, diag, n, lda); 

    let mut xbuf_coral = xbuf.clone(); 
    let mut xbuf_cblas = xbuf.clone(); 

    let aview = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xview = VectorMut::new(&mut xbuf_coral, n, incx, 0)?; 

    strmv(uplo, trans, diag, aview, xview); 

    unsafe { 
        cblas_strmv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            CBLAS_TRANSPOSE::CblasNoTrans, 
            CBLAS_DIAG::CblasNonUnit, 
            n as i32, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf_cblas.as_mut_ptr(), 
            incx as i32, 
        )
    }

    assert_close(&xbuf_coral, &xbuf_cblas, RTOL, ATOL); 
    Ok(())
}


#[test] 
fn upper_nonunit_t() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Upper; 
    let diag = CoralDiagonal::NonUnit;
    let trans = CoralTranspose::Trans; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(uplo, diag, n, lda); 

    let mut xbuf_coral = xbuf.clone(); 
    let mut xbuf_cblas = xbuf.clone(); 

    let aview = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xview = VectorMut::new(&mut xbuf_coral, n, incx, 0)?; 

    strmv(uplo, trans, diag, aview, xview); 

    unsafe { 
        cblas_strmv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            CBLAS_TRANSPOSE::CblasTrans, 
            CBLAS_DIAG::CblasNonUnit, 
            n as i32, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf_cblas.as_mut_ptr(), 
            incx as i32, 
        )
    }

    assert_close(&xbuf_coral, &xbuf_cblas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn upper_unit_n() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Upper; 
    let diag = CoralDiagonal::Unit;
    let trans = CoralTranspose::NoTrans; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(uplo, diag, n, lda); 

    let mut xbuf_coral = xbuf.clone(); 
    let mut xbuf_cblas = xbuf.clone(); 

    let aview = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xview = VectorMut::new(&mut xbuf_coral, n, incx, 0)?; 

    strmv(uplo, trans, diag, aview, xview); 

    unsafe { 
        cblas_strmv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            CBLAS_TRANSPOSE::CblasNoTrans, 
            CBLAS_DIAG::CblasUnit, 
            n as i32, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf_cblas.as_mut_ptr(), 
            incx as i32, 
        )
    }

    assert_close(&xbuf_coral, &xbuf_cblas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn upper_unit_t() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Upper; 
    let diag = CoralDiagonal::Unit;
    let trans = CoralTranspose::Trans; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(uplo, diag, n, lda); 

    let mut xbuf_coral = xbuf.clone(); 
    let mut xbuf_cblas = xbuf.clone(); 

    let aview = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xview = VectorMut::new(&mut xbuf_coral, n, incx, 0)?; 

    strmv(uplo, trans, diag, aview, xview); 

    unsafe { 
        cblas_strmv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            CBLAS_TRANSPOSE::CblasTrans, 
            CBLAS_DIAG::CblasUnit, 
            n as i32, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf_cblas.as_mut_ptr(), 
            incx as i32, 
        )
    }

    assert_close(&xbuf_coral, &xbuf_cblas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn lower_nonunit_n() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Lower; 
    let diag = CoralDiagonal::NonUnit;
    let trans = CoralTranspose::NoTrans; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(uplo, diag, n, lda); 

    let mut xbuf_coral = xbuf.clone(); 
    let mut xbuf_cblas = xbuf.clone(); 

    let aview = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xview = VectorMut::new(&mut xbuf_coral, n, incx, 0)?; 

    strmv(uplo, trans, diag, aview, xview); 

    unsafe { 
        cblas_strmv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            CBLAS_TRANSPOSE::CblasNoTrans, 
            CBLAS_DIAG::CblasNonUnit, 
            n as i32, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf_cblas.as_mut_ptr(), 
            incx as i32, 
        )
    }

    assert_close(&xbuf_coral, &xbuf_cblas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn lower_nonunit_t() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Lower; 
    let diag = CoralDiagonal::NonUnit;
    let trans = CoralTranspose::Trans; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(uplo, diag, n, lda); 

    let mut xbuf_coral = xbuf.clone(); 
    let mut xbuf_cblas = xbuf.clone(); 

    let aview = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xview = VectorMut::new(&mut xbuf_coral, n, incx, 0)?; 

    strmv(uplo, trans, diag, aview, xview); 

    unsafe { 
        cblas_strmv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            CBLAS_TRANSPOSE::CblasTrans, 
            CBLAS_DIAG::CblasNonUnit, 
            n as i32, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf_cblas.as_mut_ptr(), 
            incx as i32, 
        )
    }

    assert_close(&xbuf_coral, &xbuf_cblas, 1e-4, ATOL); 
    Ok(())
}

#[test] 
fn lower_unit_n() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Lower; 
    let diag = CoralDiagonal::Unit;
    let trans = CoralTranspose::NoTrans; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(uplo, diag, n, lda); 

    let mut xbuf_coral = xbuf.clone(); 
    let mut xbuf_cblas = xbuf.clone(); 

    let aview = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xview = VectorMut::new(&mut xbuf_coral, n, incx, 0)?; 

    strmv(uplo, trans, diag, aview, xview); 

    unsafe { 
        cblas_strmv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            CBLAS_TRANSPOSE::CblasNoTrans, 
            CBLAS_DIAG::CblasUnit, 
            n as i32, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf_cblas.as_mut_ptr(), 
            incx as i32, 
        )
    }

    assert_close(&xbuf_coral, &xbuf_cblas, 1e-4, ATOL); 
    Ok(())
}

#[test] 
fn lower_unit_t() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let uplo = CoralTriangular::Lower; 
    let diag = CoralDiagonal::Unit;
    let trans = CoralTranspose::Trans; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(uplo, diag, n, lda); 

    let mut xbuf_coral = xbuf.clone(); 
    let mut xbuf_cblas = xbuf.clone(); 

    let aview = MatrixRef::new(&abuf, n, n, lda, 0)?; 
    let xview = VectorMut::new(&mut xbuf_coral, n, incx, 0)?; 

    strmv(uplo, trans, diag, aview, xview); 

    unsafe { 
        cblas_strmv ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            CBLAS_TRANSPOSE::CblasTrans, 
            CBLAS_DIAG::CblasUnit, 
            n as i32, 
            abuf.as_ptr(), 
            lda as i32, 
            xbuf_cblas.as_mut_ptr(), 
            incx as i32, 
        )
    }

    assert_close(&xbuf_coral, &xbuf_cblas, RTOL, ATOL); 
    Ok(())
}














