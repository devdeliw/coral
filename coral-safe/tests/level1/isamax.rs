use super::common::{
    make_strided_vec, 
    CoralResult,
}; 

use blas_src as _; 
use cblas_sys::cblas_isamax; 
use coral_safe::level1::isamax; 
use coral_safe::types::VectorRef; 

#[test] 
fn unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let xbuf = make_strided_vec(n, incx); 
    let xvec = VectorRef::new(&xbuf, n, incx, 0)?; 

    let coral_val = isamax(xvec); 
    let cblas_val = unsafe { 
        cblas_isamax ( 
            n as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
        ) as usize
    }; 

    assert_eq!(coral_val, cblas_val); 
    Ok(())
}

#[test] 
fn strided() -> CoralResult { 
    let n = 1024; 
    let incx = 3; 
    let xbuf = make_strided_vec(n, incx); 
    let xvec = VectorRef::new(&xbuf, n, incx, 0)?; 

    let coral_val = isamax(xvec); 
    let cblas_val = unsafe { 
        cblas_isamax ( 
            n as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
        ) as usize
    }; 

    assert_eq!(coral_val, cblas_val); 
    Ok(())
}

#[test] 
fn n_zero() -> CoralResult { 
    let n = 0; 
    let incx = 1; 
    let xbuf = make_strided_vec(n, incx); 
    let xvec = VectorRef::new(&xbuf, n, incx, 0)?; 

    let coral_val = isamax(xvec); 
    let cblas_val = unsafe { 
        cblas_isamax ( 
            n as i32, 
            xbuf.as_ptr(), 
            incx as i32, 
        ) as usize
    }; 

    assert_eq!(coral_val, cblas_val); 
    Ok(())
}


