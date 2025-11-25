use super::common::{
    make_strided_vec, 
    assert_close, 
    CoralResult,
    RTOL, 
    ATOL, 
}; 

use blas_src as _; 
use cblas_sys::cblas_snrm2; 
use coral::level1::snrm2; 
use coral::types::VectorRef; 

#[test]
fn unit_stride() -> CoralResult { 
    let n = 1024; 
    let xbuf = make_strided_vec(n, 1);
    let xvec = VectorRef::new ( 
        &xbuf, 
        n, 
        1, 
        0
    )?; 

    let coral_val = snrm2(xvec); 
    let cblas_val = unsafe { 
        cblas_snrm2 ( 
            n as i32, 
            xbuf.as_ptr(), 
            1,
        )
    };

    assert_close(&[coral_val], &[cblas_val], RTOL, ATOL);
    Ok(()) 
}

#[test]
fn strided() -> CoralResult { 
    let n = 1024; 
    let incx = 3; 
    let xbuf = make_strided_vec(n, incx); 
    let xvec = VectorRef::new ( 
        &xbuf, 
        n, 
        incx, 
        0, 
    )?; 

    let coral_val = snrm2(xvec); 
    let cblas_val = unsafe { 
        cblas_snrm2 ( 
            n as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
        )
    };

    assert_close(&[coral_val], &[cblas_val], RTOL, ATOL); 
    Ok(())
}

#[test]
fn n_zero() -> CoralResult { 
    let n = 0; 
    let incx = 1; 
    let xbuf = make_strided_vec(n, incx); 
    let xvec = VectorRef::new ( 
        &xbuf, 
        n, 
        incx, 
        0, 
    )?; 

    let coral_val = snrm2(xvec); 
    let cblas_val = unsafe { 
        cblas_snrm2 ( 
            n as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
        )
    };

    assert_close(&[coral_val], &[cblas_val], RTOL, ATOL); 
    Ok(())
}

