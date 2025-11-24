use crate::level3::packers::{MR, NR, a_buf_len, b_buf_len};
use crate::level3::microkernel::{
    edge::edge,
    block::{
        mrxnr_beta0,
        mrxnr_beta1,
        mrxnr_betax,
    },
};

#[inline(always)]
pub(crate) fn macrokernel(
    mc: usize,
    nc: usize,
    kc: usize,
    alpha: f32,
    beta_panel: f32,
    a_pack: &[f32],   
    b_pack: &[f32],   
    c_base: &mut [f32],
    ldc: usize,
) {
    let np = nc.div_ceil(NR); 
    let mp = mc.div_ceil(MR); 
    debug_assert!(a_pack.len() >= a_buf_len(mc, kc));
    debug_assert!(b_pack.len() >= b_buf_len(kc, nc));

    if mc > 0 && nc > 0 {
        debug_assert!(c_base.len() >= (nc - 1) * ldc + mc);
    }

    for jp in 0..np {
        let nr = std::cmp::min(NR, nc.saturating_sub(jp * NR));
        if nr == 0 {
            break;
        }

        let bp_offset = jp * kc * NR;
        let b_block = &b_pack[bp_offset .. bp_offset + kc * NR];

        for ip in 0..mp {
            let mr = std::cmp::min(MR, mc.saturating_sub(ip * MR));
            if mr == 0 {
                break;
            }

            let ap_offset = ip * kc * MR;
            let a_block = &a_pack[ap_offset .. ap_offset + kc * MR];

            let c_offset = ip * MR + (jp * NR) * ldc;
            let c_block = &mut c_base[c_offset..];

            if mr == MR && nr == NR {
                // full tile, optimized
                if beta_panel == 0.0 {
                    mrxnr_beta0(kc, a_block, b_block, c_block, ldc, alpha);
                } else if beta_panel == 1.0 {
                    mrxnr_beta1(kc, a_block, b_block, c_block, ldc, alpha);
                } else {
                    mrxnr_betax(kc, a_block, b_block, c_block, ldc, alpha, beta_panel);
                }
            } else {
                // scalar edge 
                edge (
                    mr,
                    nr,
                    kc,
                    a_block,
                    b_block,
                    c_block,
                    ldc,
                    alpha,
                    beta_panel,
                );
            }
        }
    }
}

