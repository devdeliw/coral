use crate::level3::c64_packers::{MR, NR};
use crate::level3::microkernel::{
    c64_edge::c64_edge,
    c64_mrxnr::{
        c64_mrxnr_beta0,
        c64_mrxnr_beta1,
        c64_mrxnr_betax,
        Complex64,
    },
};

#[inline(always)]
pub(crate) fn macro_kernel(
    mc         : usize,
    nc         : usize,
    kc         : usize,
    alpha      : Complex64,
    beta_panel : Complex64,
    a_pack     : *const f64,
    b_pack     : *const f64,
    c_base     : *mut f64,
    ldc        : usize,           
) {
    unsafe {
        let np = (nc + NR - 1) / NR;
        let mp = (mc + MR - 1) / MR;

        let a_panel_stride = kc * (2 * MR);
        let b_panel_stride = kc * (2 * NR);

        let beta_is_zero = beta_panel.re == 0.0 && beta_panel.im == 0.0;
        let beta_is_one  = beta_panel.re == 1.0 && beta_panel.im == 0.0;

        for jp in 0..np {
            let nr = core::cmp::min(NR, nc - jp * NR);
            let bp = b_pack.add(jp * b_panel_stride);

            for ip in 0..mp {
                let mr = core::cmp::min(MR, mc - ip * MR);
                let ap = a_pack.add(ip * a_panel_stride);

                let cptr = c_base.add(2 * (ip * MR + (jp * NR) * ldc));

                if mr == MR && nr == NR {
                    if beta_is_zero {

                        c64_mrxnr_beta0(kc, ap, bp, cptr, ldc, alpha);

                    } else if beta_is_one {

                        c64_mrxnr_beta1(kc, ap, bp, cptr, ldc, alpha);

                    } else {

                        c64_mrxnr_betax(kc, ap, bp, cptr, ldc, alpha, beta_panel);

                    }
                } else {

                    c64_edge(mr, nr, kc, ap, bp, cptr, ldc, alpha, beta_panel);
                }
            }
        }
    }
}

