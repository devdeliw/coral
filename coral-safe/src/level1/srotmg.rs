//! Level 1 [`?ROTMG`](https://www.netlib.org/lapack/explore-html/d3/dd5/group__rotmg.html)
//! routine in single precision
//!
//! Generates transformation that zeros the second component of the 2-vector 
//!
//! //! \\[
//! \begin{bmatrix}
//! \sqrt{sd_1}\\, sx_1 \\\\
//! \sqrt{sd_2}\\, sy_1
//! \end{bmatrix}
//! \\]
//!
//! # Author 
//! Deval Deliwala 


/// Generates parameter array `param` that defines a modified Givens rotation 
/// that zeros the 2-vector [`sqrt(sd_1) sx_1, sqrt(sd_2)sy_1`]. 
///
/// The rotation operator is encoded in `param` with flag `param[0]`. 
///
/// Arguments: 
/// * `sd1`: `&mut` [f32] - input/output scalar, updated scaling factor for first component 
/// * `sd2`: `&mut` [f32] - input/output scalar, updated scaling factor for second component 
/// * `sx1`: `&mut` [f32] - input/output scalar, updated first vector componnet 
/// * `sy1`: [f32] - input scalar, second vector component; not modified. 
/// * `param`: `&mut [f32; 5] - output array of 5 elements defining the modified Givesn rot. 
///
/// Returns: 
/// Nothing. overwrites `sd1`, `sd2`, `sx1`, and fills `param`.
#[inline] 
pub fn srotmg ( 
    sd1: &mut f32, 
    sd2: &mut f32, 
    sx1: &mut f32, 
    sy1: f32, 
    param: &mut [f32; 5]
) { 
    const GAM: f32 = 4096.0; 
    const GAMSQ: f32 = GAM * GAM; 
    const RGAMSQ: f32 = 1.0 / GAMSQ; 

    let mut sflag: f32;
    let mut sh11: f32 = 0.0; 
    let mut sh12: f32 = 0.0; 
    let mut sh21: f32 = 0.0; 
    let mut sh22: f32 = 0.0; 

    // undefined, kill 
    if *sd1 < 0.0 { 
        sflag = -1.0; 
        *sd1 = 0.0; 
        *sd2 = 0.0; 
        *sx1 = 0.0; 
   } else { 
        let sp2 = *sd2 * sy1; 

        if sp2 == 0.0 { 
            param[0] = -2.0; 
            return; 
        }

        let sp1 = *sd1 * *sx1; 
        let sq2 = sp2 * sy1; 
        let sq1 = sp1 * *sx1; 

        if sq1.abs() > sq2.abs() { 
            sh21 = -sy1 / *sx1; 
            sh12 = sp2 / sp1; 

            let su = 1.0 - sh12 * sh21; 

            // undefined, kill 
            if su <= 0.0 { 
                sflag = -1.0;
                *sd1  = 0.0;
                *sd2  = 0.0;
                *sx1  = 0.0;
                sh11  = 0.0; 
                sh12  = 0.0; 
                sh21  = 0.0; 
                sh22  = 0.0;
            } else {
                sflag = 0.0;
                *sd1 /= su;
                *sd2 /= su;
                *sx1 *= su;
            }
        } else {
            
            // undefined, kill 
            if sq2 < 0.0 {
                sflag = -1.0;
                *sd1  = 0.0;
                *sd2  = 0.0;
                *sx1  = 0.0;
                sh11  = 0.0; 
                sh12  = 0.0; 
                sh21  = 0.0; 
                sh22  = 0.0;
            } else {
                sflag = 1.0;
                sh11  = sp1 / sp2;
                sh22  = *sx1 / sy1;

                let su = 1.0 + sh11 * sh22;
                let stemp = *sd2 / su;
                *sd2 = *sd1 / su;
                *sd1 = stemp;
                *sx1 = sy1 * su;

            }
        }

        if *sd1 != 0.0 {
            while *sd1 <= RGAMSQ || *sd1 >= GAMSQ {
                if sflag == 0.0 {
                    sh11  = 1.0;
                    sh22  = 1.0;
                    sflag = -1.0;
                } else {
                    sh21  = -1.0;
                    sh12  = 1.0;
                    sflag = -1.0;
                }

                if *sd1  <= RGAMSQ {
                    *sd1 *= GAMSQ;
                    *sx1 /= GAM;
                    sh11 /= GAM;
                    sh12 /= GAM;
                } else {
                    *sd1 /= GAMSQ;
                    *sx1 *= GAM;
                    sh11 *= GAM;
                    sh12 *= GAM;
                }
            }
        }

        if *sd2 != 0.0 {
            while sd2.abs() <= RGAMSQ || sd2.abs() >= GAMSQ {
                if sflag == 0.0 {
                    sh11  = 1.0;
                    sh22  = 1.0;
                    sflag = -1.0;
                } else {
                    sh21  = -1.0;
                    sh12  = 1.0;
                    sflag = -1.0;
                }

                if sd2.abs() <= RGAMSQ {
                    *sd2 *= GAMSQ;
                    sh21 /= GAM;
                    sh22 /= GAM;
                } else {
                    *sd2 /= GAMSQ;
                    sh21 *= GAM;
                    sh22 *= GAM;
                }
            }
        }
    }

    param[1] = 0.0;
    param[2] = 0.0;
    param[3] = 0.0;
    param[4] = 0.0;

    if sflag < 0.0 {
        param[1] = sh11;
        param[2] = sh21;
        param[3] = sh12;
        param[4] = sh22;
    } else if sflag == 0.0 {
        param[2] = sh21;
        param[3] = sh12;
    } else {
        param[1] = sh11;
        param[4] = sh22;
    }

    param[0] = sflag;
}
