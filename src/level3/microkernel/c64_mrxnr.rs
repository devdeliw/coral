use crate::level3::c64_packers::{MR, NR};
use core::arch::aarch64::{
    float64x2_t, float64x2x2_t, vdupq_n_f64, vld1q_f64, vld2q_f64, vfmaq_laneq_f64,
    vfmsq_laneq_f64, vmulq_n_f64, vst2q_f64,
};

#[repr(C)] // HFA
#[derive(Copy, Clone)]
pub(crate) struct Complex64 {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

#[inline(always)]
fn load_a_rowpairs(ap: *const f64) -> (float64x2_t, float64x2_t) {
    debug_assert_eq!(MR, 2);
    unsafe {
        // A per k-step: 
        // [a_re(0..MR), a_im(0..MR)]
        let a_re = vld1q_f64(ap.add(0));
        let a_im = vld1q_f64(ap.add(MR));
        (a_re, a_im)
    }
}

#[inline(always)]
fn load_b_colpairs(
    bp: *const f64,
) -> (float64x2_t, float64x2_t, float64x2_t, float64x2_t) {
    debug_assert_eq!(NR, 4);
    unsafe {
        // B per k-step: 
        // [b_re(0..NR), b_im(0..NR)]
        // split into two lanes for re/im
        let br01 = vld1q_f64(bp.add(0));
        let br23 = vld1q_f64(bp.add(2));
        let bi01 = vld1q_f64(bp.add(NR + 0));
        let bi23 = vld1q_f64(bp.add(NR + 2));
        (br01, br23, bi01, bi23)
    }
}

#[inline(always)]
fn store_col2(colp: *mut f64, v_re: float64x2_t, v_im: float64x2_t, ldc: usize) {
    let _ = ldc;
    unsafe {
        let pair = float64x2x2_t(v_re, v_im);
        vst2q_f64(colp, pair);
    }
}

#[inline(always)]
fn load_col2(colp: *const f64, ldc: usize) -> (float64x2_t, float64x2_t) {
    let _ = ldc;
    unsafe {
        let pair = vld2q_f64(colp);
        (pair.0, pair.1)
    }
}

#[inline(always)]
fn kstep_accumulate_laneq(
    acc_re : &mut [float64x2_t; NR],
    acc_im : &mut [float64x2_t; NR],
    a_re   : float64x2_t,
    a_im   : float64x2_t,
    br01   : float64x2_t,
    br23   : float64x2_t,
    bi01   : float64x2_t,
    bi23   : float64x2_t,
) {
    unsafe {
        // cols 0..1; lanes from br01/bi01
        acc_re[0] = vfmaq_laneq_f64(acc_re[0], a_re, br01, 0);
        acc_re[0] = vfmsq_laneq_f64(acc_re[0], a_im, bi01, 0);
        acc_im[0] = vfmaq_laneq_f64(acc_im[0], a_re, bi01, 0);
        acc_im[0] = vfmaq_laneq_f64(acc_im[0], a_im, br01, 0);

        acc_re[1] = vfmaq_laneq_f64(acc_re[1], a_re, br01, 1);
        acc_re[1] = vfmsq_laneq_f64(acc_re[1], a_im, bi01, 1);
        acc_im[1] = vfmaq_laneq_f64(acc_im[1], a_re, bi01, 1);
        acc_im[1] = vfmaq_laneq_f64(acc_im[1], a_im, br01, 1);

        // cols 2..3; lanes from br23/bi23
        acc_re[2] = vfmaq_laneq_f64(acc_re[2], a_re, br23, 0);
        acc_re[2] = vfmsq_laneq_f64(acc_re[2], a_im, bi23, 0);
        acc_im[2] = vfmaq_laneq_f64(acc_im[2], a_re, bi23, 0);
        acc_im[2] = vfmaq_laneq_f64(acc_im[2], a_im, br23, 0);

        acc_re[3] = vfmaq_laneq_f64(acc_re[3], a_re, br23, 1);
        acc_re[3] = vfmsq_laneq_f64(acc_re[3], a_im, bi23, 1);
        acc_im[3] = vfmaq_laneq_f64(acc_im[3], a_re, bi23, 1);
        acc_im[3] = vfmaq_laneq_f64(acc_im[3], a_im, br23, 1);
    }
}

#[inline(always)]
fn apply_alpha(
    acc_re : float64x2_t,
    acc_im : float64x2_t,
    alpha  : Complex64,
) -> (float64x2_t, float64x2_t) {
    unsafe {
        // (acc_re*ar - acc_im*ai,  acc_re*ai + acc_im*ar)
        let ar = alpha.re;
        let ai = alpha.im;

        let mut out_re = vmulq_n_f64(acc_re, ar);
        out_re = vfmsq_laneq_f64(out_re, acc_im, vdupq_n_f64(ai), 0);
        let mut out_im = vmulq_n_f64(acc_im, ar);
        out_im = vfmaq_laneq_f64(out_im, acc_re, vdupq_n_f64(ai), 0);

        (out_re, out_im)
    }
}

#[inline(always)]
fn apply_beta(
    c_re : float64x2_t,
    c_im : float64x2_t,
    beta : Complex64,
) -> (float64x2_t, float64x2_t) {
    unsafe {
        // (c_re*br - c_im*bi,  c_re*bi + c_im*br)
        let br = beta.re;
        let bi = beta.im;

        let mut out_re = vmulq_n_f64(c_re, br);
        out_re = vfmsq_laneq_f64(out_re, c_im, vdupq_n_f64(bi), 0);
        let mut out_im = vmulq_n_f64(c_im, br);
        out_im = vfmaq_laneq_f64(out_im, c_re, vdupq_n_f64(bi), 0);

        (out_re, out_im)
    }
}

// beta = 0

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn c64_mrxnr_beta0(
    kc    : usize,
    a     : *const f64, // packed A; per k-step [MR re | MR im]
    b     : *const f64, // packed B; per k-step [NR re | NR im]
    c     : *mut f64,   
    ldc   : usize,      // complex elems
    alpha : Complex64,
) {
    debug_assert_eq!(MR, 2);
    debug_assert_eq!(NR, 4);

    unsafe {
        let mut acc_re: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];
        let mut acc_im: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            // k
            let (br01, br23, bi01, bi23) = load_b_colpairs(bp);
            let (a_re, a_im)             = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc_re, 
                &mut acc_im,
                a_re,
                a_im, 
                br01, 
                br23, 
                bi01,
                bi23
            );

            // k+1
            let (br01n, br23n, bi01n, bi23n) = load_b_colpairs(bp.add(2 * NR));
            let (a_ren, a_imn)               = load_a_rowpairs(ap.add(2 * MR));
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im, 
                a_ren, 
                a_imn,
                br01n, 
                br23n, 
                bi01n, 
                bi23n
            );

            ap = ap.add(4 * MR);
            bp = bp.add(4 * NR);
        }

        if kc & 1 != 0 {
            let (br01, br23, bi01, bi23) = load_b_colpairs(bp);
            let (a_re, a_im)             = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im, 
                a_re,
                a_im, 
                br01, 
                br23, 
                bi01, 
                bi23
            );
        }

        for j in 0..NR {
            let colp = c.add(2 * j * ldc);
            let (out_re, out_im) = apply_alpha(acc_re[j], acc_im[j], alpha);
            store_col2(colp, out_re, out_im, ldc);
        }
    }
}

//  beta = 1

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn c64_mrxnr_beta1(
    kc    : usize,
    a     : *const f64,
    b     : *const f64,
    c     : *mut f64,
    ldc   : usize,
    alpha : Complex64,
) {
    debug_assert_eq!(MR, 2);
    debug_assert_eq!(NR, 4);

    unsafe {
        let mut acc_re: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];
        let mut acc_im: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            let (br01, br23, bi01, bi23) = load_b_colpairs(bp);
            let (a_re, a_im)             = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc_re, 
                &mut acc_im, 
                a_re, 
                a_im,
                br01, 
                br23, 
                bi01, 
                bi23
            );

            let (br01n, br23n, bi01n, bi23n) = load_b_colpairs(bp.add(2 * NR));
            let (a_ren, a_imn)               = load_a_rowpairs(ap.add(2 * MR));
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im,
                a_ren, 
                a_imn, 
                br01n, 
                br23n,
                bi01n,
                bi23n
            );

            ap = ap.add(4 * MR);
            bp = bp.add(4 * NR);
        }
        if kc & 1 != 0 {
            let (br01, br23, bi01, bi23) = load_b_colpairs(bp);
            let (a_re, a_im)             = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc_re, 
                &mut acc_im, 
                a_re, 
                a_im, 
                br01, 
                br23, 
                bi01, 
                bi23
            );
        }

        let one = vdupq_n_f64(1.0);
        for j in 0..NR {
            let colp = c.add(2 * j * ldc);
            let (mut c_re, mut c_im) = load_col2(colp, ldc);
            let (add_re, add_im)     = apply_alpha(acc_re[j], acc_im[j], alpha);

            c_re = vfmaq_laneq_f64(c_re, add_re, one, 0);
            c_im = vfmaq_laneq_f64(c_im, add_im, one, 0);

            store_col2(colp, c_re, c_im, ldc);
        }
    }
}

// general beta

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn c64_mrxnr_betax(
    kc    : usize,
    a     : *const f64,
    b     : *const f64,
    c     : *mut f64,
    ldc   : usize,
    alpha : Complex64,
    beta  : Complex64,
) {
    debug_assert_eq!(MR, 2);
    debug_assert_eq!(NR, 4);

    unsafe {
        let mut acc_re: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];
        let mut acc_im: [float64x2_t; NR] = [vdupq_n_f64(0.0); NR];

        let mut ap = a;
        let mut bp = b;

        let k2 = kc / 2;
        for _ in 0..k2 {
            let (br01, br23, bi01, bi23) = load_b_colpairs(bp);
            let (a_re, a_im)             = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im,
                a_re,
                a_im, 
                br01,
                br23,
                bi01,
                bi23
            );

            let (br01n, br23n, bi01n, bi23n) = load_b_colpairs(bp.add(2 * NR));
            let (a_ren, a_imn)               = load_a_rowpairs(ap.add(2 * MR));
            kstep_accumulate_laneq(
                &mut acc_re,
                &mut acc_im,
                a_ren, 
                a_imn,
                br01n,
                br23n, 
                bi01n, 
                bi23n
            );

            ap = ap.add(4 * MR);
            bp = bp.add(4 * NR);
        }
        if kc & 1 != 0 {
            let (br01, br23, bi01, bi23) = load_b_colpairs(bp);
            let (a_re, a_im)             = load_a_rowpairs(ap);
            kstep_accumulate_laneq(
                &mut acc_re, 
                &mut acc_im, 
                a_re, 
                a_im, 
                br01, 
                br23, 
                bi01, 
                bi23
            );
        }

        let one = vdupq_n_f64(1.0);
        for j in 0..NR {
            let colp = c.add(2 * j * ldc);

            let (c_re0, c_im0) = load_col2(colp, ldc);
            let (mut c_re, mut c_im) = apply_beta(c_re0, c_im0, beta);
            let (add_re, add_im)     = apply_alpha(acc_re[j], acc_im[j], alpha);

            c_re = vfmaq_laneq_f64(c_re, add_re, one, 0);
            c_im = vfmaq_laneq_f64(c_im, add_im, one, 0);

            store_col2(colp, c_re, c_im, ldc);
        }
    }
}

