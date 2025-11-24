use super::common::{
    make_strided_mat, 
    make_strided_vec, 
    assert_close, 
    CoralResult, 
    ATOL, 
    RTOL, 
}; 

use blas_src as _; 
use cblas_sys::{cblas_ssyr, CBLAS_LAYOUT, CBLAS_UPLO}; 
use coral_safe::level2::ssyr; 
use coral_safe::types::{MatrixMut, VectorRef, CoralTriangular}; 


#[test] 
fn upper_unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n + 32; 
    
    let alpha = 3.14159265358979; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx); 

    let mut asafe = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = VectorRef::new(&xbuf, n, incx, 0)?; 
    let aview = MatrixMut::new(&mut asafe, n, n, lda, 0)?;

    ssyr(CoralTriangular::Upper, alpha, aview, xview);
    unsafe { 
        cblas_ssyr ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            n as i32, 
            alpha, 
            xbuf.as_ptr(), 
            incx as i32, 
            ablas.as_mut_ptr(), 
            lda as i32, 
        )
    }

    assert_close(&asafe, &ablas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn lower_unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 
    
    let alpha = 3.14159265358979; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx); 

    let mut asafe = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = VectorRef::new(&xbuf, n, incx, 0)?; 
    let aview = MatrixMut::new(&mut asafe, n, n, lda, 0)?;

    ssyr(CoralTriangular::Lower, alpha, aview, xview);
    unsafe { 
        cblas_ssyr ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            n as i32, 
            alpha, 
            xbuf.as_ptr(), 
            incx as i32, 
            ablas.as_mut_ptr(), 
            lda as i32, 
        )
    }

    assert_close(&asafe, &ablas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn upper_strided() -> CoralResult { 
    let n = 1024; 
    let incx = 5; 
    let lda = n + 64; 
    
    let alpha = 3.14159265358979; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx); 

    let mut asafe = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = VectorRef::new(&xbuf, n, incx, 0)?; 
    let aview = MatrixMut::new(&mut asafe, n, n, lda, 0)?;

    ssyr(CoralTriangular::Upper, alpha, aview, xview);
    unsafe { 
        cblas_ssyr ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            n as i32, 
            alpha, 
            xbuf.as_ptr(), 
            incx as i32, 
            ablas.as_mut_ptr(), 
            lda as i32, 
        )
    }

    assert_close(&asafe, &ablas, RTOL, ATOL); 
    Ok(())
}

#[test] 
fn lower_strided() -> CoralResult { 
    let n = 1024; 
    let incx = 3; 
    let lda = n + 32; 
    
    let alpha = 3.14159265358979; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx); 

    let mut asafe = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = VectorRef::new(&xbuf, n, incx, 0)?; 
    let aview = MatrixMut::new(&mut asafe, n, n, lda, 0)?;

    ssyr(CoralTriangular::Lower, alpha, aview, xview);
    unsafe { 
        cblas_ssyr ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            n as i32, 
            alpha, 
            xbuf.as_ptr(), 
            incx as i32, 
            ablas.as_mut_ptr(), 
            lda as i32, 
        )
    }

    assert_close(&asafe, &ablas, RTOL, ATOL); 
    Ok(())
}


