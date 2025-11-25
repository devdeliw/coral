use super::common::{
    make_strided_mat, 
    make_strided_vec, 
    assert_close, 
    CoralResult, 
    ATOL, 
    RTOL, 
}; 

use blas_src as _; 
use cblas_sys::{cblas_ssyr2, CBLAS_LAYOUT, CBLAS_UPLO}; 
use coral::level2::ssyr2; 
use coral::types::{MatrixMut, VectorRef, CoralTriangular}; 


#[test] 
fn upper_unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1;
    let incy = 1; 
    let lda = n + 32; 
    
    let alpha = 3.14159265358979; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx);
    let ybuf = make_strided_vec(n, incy); 

    let mut asafe = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yview = VectorRef::new(&ybuf, n, incy, 0)?; 
    let aview = MatrixMut::new(&mut asafe, n, n, lda, 0)?;

    ssyr2(CoralTriangular::Upper, alpha, aview, xview, yview);
    unsafe { 
        cblas_ssyr2 ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            n as i32, 
            alpha, 
            xbuf.as_ptr(), 
            incx as i32, 
            ybuf.as_ptr(), 
            incy as i32, 
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
    let incy = 1; 
    let lda = n; 
    
    let alpha = 3.14159265358979; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx);
    let ybuf = make_strided_vec(n, incy); 

    let mut asafe = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yview = VectorRef::new(&ybuf, n, incy, 0)?; 
    let aview = MatrixMut::new(&mut asafe, n, n, lda, 0)?;

    ssyr2(CoralTriangular::Lower, alpha, aview, xview, yview);
    unsafe { 
        cblas_ssyr2 ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            n as i32, 
            alpha, 
            xbuf.as_ptr(), 
            incx as i32, 
            ybuf.as_ptr(), 
            incy as i32, 
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
    let incx = 2;
    let incy = 5; 
    let lda = n + 32; 
    
    let alpha = 3.14159265358979; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx);
    let ybuf = make_strided_vec(n, incy); 

    let mut asafe = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yview = VectorRef::new(&ybuf, n, incy, 0)?; 
    let aview = MatrixMut::new(&mut asafe, n, n, lda, 0)?;

    ssyr2(CoralTriangular::Upper, alpha, aview, xview, yview);
    unsafe { 
        cblas_ssyr2 ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasUpper, 
            n as i32, 
            alpha, 
            xbuf.as_ptr(), 
            incx as i32, 
            ybuf.as_ptr(), 
            incy as i32, 
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
    let incx = 2;
    let incy = 3; 
    let lda = n + 32; 
    
    let alpha = 3.14159265358979; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx);
    let ybuf = make_strided_vec(n, incy); 

    let mut asafe = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = VectorRef::new(&xbuf, n, incx, 0)?; 
    let yview = VectorRef::new(&ybuf, n, incy, 0)?; 
    let aview = MatrixMut::new(&mut asafe, n, n, lda, 0)?;

    ssyr2(CoralTriangular::Lower, alpha, aview, xview, yview);
    unsafe { 
        cblas_ssyr2 ( 
            CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_UPLO::CblasLower, 
            n as i32, 
            alpha, 
            xbuf.as_ptr(), 
            incx as i32, 
            ybuf.as_ptr(), 
            incy as i32, 
            ablas.as_mut_ptr(), 
            lda as i32, 
        )
    }

    assert_close(&asafe, &ablas, RTOL, ATOL); 
    Ok(())
}
