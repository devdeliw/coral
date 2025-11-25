use super::common::{
    make_strided_vec, 
    assert_close, 
    CoralResult,
    RTOL, 
    ATOL, 
}; 

use blas_src as _; 
use cblas_sys::cblas_srotm; 
use coral::level1::srotm; 
use coral::types::VectorMut; 

const FLAGS: [f32; 4] = [-2.0, -1.0, 0.0, 1.0];
const PARAM: [f32; 4] = [0.9, -0.3, -0.4, 1.1]; 

#[test]
fn unit_stride() -> CoralResult { 
    let n = 1024; 
    let incx = 1; 
    let incy = 1; 
    
    let mut xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 
    let mut xcblas = xbuf.clone(); 
    let mut ycblas = ybuf.clone();
    let [p1, p2, p3, p4] = PARAM; 

    for &flag in FLAGS.iter() {
        let xvec = VectorMut::new(&mut xbuf, n, incx, 0)?;
        let yvec = VectorMut::new(&mut ybuf, n, incy, 0)?;
        let params: [f32; 5] = [flag, p1, p2, p3, p4]; 
        
        srotm(xvec, yvec, &params); 
        unsafe { 
            cblas_srotm ( 
                n as i32, 
                xcblas.as_mut_ptr(), 
                incx as i32, 
                ycblas.as_mut_ptr(), 
                incy as i32, 
                params.as_ptr(), 
            );
        }

        assert_close(&xbuf, &xcblas, RTOL, ATOL); 
        assert_close(&ybuf, &ycblas, RTOL, ATOL);
    }

    Ok(())
}

#[test]
fn strided() -> CoralResult { 
    let n = 1024; 
    let incx = 3; 
    let incy = 5; 
    
    let mut xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 
    let mut xcblas = xbuf.clone(); 
    let mut ycblas = ybuf.clone();
    let [p1, p2, p3, p4] = PARAM; 

    for &flag in FLAGS.iter() {
        let xvec = VectorMut::new(&mut xbuf, n, incx, 0)?;
        let yvec = VectorMut::new(&mut ybuf, n, incy, 0)?;
        let params: [f32; 5] = [flag, p1, p2, p3, p4]; 
        
        srotm(xvec, yvec, &params); 
        unsafe { 
            cblas_srotm ( 
                n as i32, 
                xcblas.as_mut_ptr(), 
                incx as i32, 
                ycblas.as_mut_ptr(), 
                incy as i32, 
                params.as_ptr(), 
            );
        }

        assert_close(&xbuf, &xcblas, RTOL, ATOL); 
        assert_close(&ybuf, &ycblas, RTOL, ATOL);
    }

    Ok(())
}

#[test]
fn n_zero() -> CoralResult { 
    let n = 0; 
    let incx = 1; 
    let incy = 1; 
    
    let mut xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 
    let mut xcblas = xbuf.clone(); 
    let mut ycblas = ybuf.clone();
    let [p1, p2, p3, p4] = PARAM; 

    for &flag in FLAGS.iter() {
        let xvec = VectorMut::new(&mut xbuf, n, incx, 0)?;
        let yvec = VectorMut::new(&mut ybuf, n, incy, 0)?;
        let params: [f32; 5] = [flag, p1, p2, p3, p4];         

        srotm(xvec, yvec, &params); 
        unsafe { 
            cblas_srotm ( 
                n as i32, 
                xcblas.as_mut_ptr(), 
                incx as i32, 
                ycblas.as_mut_ptr(), 
                incy as i32, 
                params.as_ptr(), 
            );
        }

        assert_close(&xbuf, &xcblas, RTOL, ATOL); 
        assert_close(&ybuf, &ycblas, RTOL, ATOL);
    }

    Ok(())
}



