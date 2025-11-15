use super::common::{
    assert_close, 
    CoralResult,
    RTOL, 
    ATOL, 
}; 

use blas_src as _; 
use cblas_sys::cblas_srotg; 
use coral_safe::level1::srotg; 

#[test]
fn matches() -> CoralResult { 
    let mut a_coral = 0.5; 
    let mut b_coral = -1.25; 
    let mut c_coral = 0.0; 
    let mut s_coral = 0.0; 

    let mut a_cblas = a_coral; 
    let mut b_cblas = b_coral; 
    let mut c_cblas = c_coral; 
    let mut s_cblas = s_coral; 

    srotg(&mut a_coral, &mut b_coral, &mut c_coral, &mut s_coral); 
    unsafe { 
        cblas_srotg ( 
            &mut a_cblas as *mut f32, 
            &mut b_cblas as *mut f32, 
            &mut c_cblas as *mut f32, 
            &mut s_cblas as *mut f32, 
        );
    }

    assert_close(&[a_coral], &[a_cblas], RTOL, ATOL);
    assert_close(&[b_coral], &[b_cblas], RTOL, ATOL);
    assert_close(&[c_coral], &[c_cblas], RTOL, ATOL);
    assert_close(&[s_coral], &[s_cblas], RTOL, ATOL);

    Ok(())
}
