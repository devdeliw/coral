use super::common::{
    assert_close, 
    CoralResult,
    RTOL, 
    ATOL, 
}; 

use blas_src as _; 
use cblas_sys::cblas_srotmg; 
use coral_safe::level1::srotmg;

#[test]
fn matches() -> CoralResult { 
    let mut d1_coral = 1.5; 
    let mut d2_coral = 2.0; 
    let mut x1_coral = -0.75; 
    let y1_coral = 0.5; 
    let mut param_coral = [0.0; 5];

    let mut d1_cblas = d1_coral; 
    let mut d2_cblas = d2_coral;
    let mut x1_cblas = x1_coral;
    let y1_cblas = y1_coral; 
    let mut param_cblas = [0.0; 5]; 

    srotmg(&mut d1_coral, &mut d2_coral, &mut x1_coral, y1_coral, &mut param_coral);
    unsafe { 
        cblas_srotmg ( 
            &mut d1_cblas as *mut f32, 
            &mut d2_cblas as *mut f32, 
            &mut x1_cblas as *mut f32, 
            y1_cblas, 
            param_cblas.as_mut_ptr(),
        ); 
    }

    assert_close(&[d1_coral], &[d1_cblas], RTOL, ATOL);
    assert_close(&[d2_coral], &[d2_cblas], RTOL, ATOL); 
    assert_close(&[x1_coral], &[x1_cblas], RTOL, ATOL); 
    assert_close(&param_coral, &param_cblas, RTOL, ATOL); 

    Ok(())
}
