pub const MR: usize = 8; 
pub const NR: usize = 12; 

#[inline(always)] 
const fn round_up(x: usize, b: usize) -> usize { 
    x.div_ceil(b) * b 
}

pub(crate) fn a_buf_len(mc: usize, kc: usize) -> usize { 
    round_up(mc, MR) * kc 
}

pub(crate) fn b_buf_len(kc: usize, nc: usize) -> usize { 
    round_up(nc, NR) * kc
}


// A side 

/// pack one `MR x k` micropanel from A 
#[inline(always)] 
fn pack_a_mrxk ( 
    k: usize, 
    a: &[f32], 
    lda: usize, 
    a_offset: usize, 
    dst: &mut [f32], 
) { 
    debug_assert!(dst.len() >= MR * k); 

    let mut ap = a_offset; 
    let mut dp = 0; 

    for _ in 0..k {
        // copy MR contiguous rows in this column 
        let src = &a[ap .. ap + MR]; 
        let dst_chunk = &mut dst[dp .. dp + MR]; 
        dst_chunk.copy_from_slice(src); 

        ap += lda; 
        dp += MR; 
    }
}

//. pack the `mr_tail x k` micropanel from A 
#[inline(always)] 
fn pack_a_mrxk_tail ( 
    k: usize, 
    a: &[f32], 
    lda: usize, 
    a_offset: usize, 
    mr_tail: usize, 
    dst: &mut [f32] 
) { 
    debug_assert!(mr_tail > 0 && mr_tail < MR); 
    debug_assert!(dst.len() >= MR * k); 

    let mut ap = a_offset; 
    let mut dp = 0; 

    for _ in 0..k { 
        let dst_chunk = &mut dst[dp .. dp + MR]; 

        let src = &a[ap .. ap + mr_tail];
        dst_chunk[..mr_tail].copy_from_slice(src);

        for x in &mut dst_chunk[mr_tail..MR] {
            *x = 0.0;
        }

        ap += lda;
        dp += MR;
    }
}

/// pack one `MR x k` micro-panel from A^T 
#[inline(always)]
fn pack_at_mrxk(
    k: usize,
    a: &[f32],
    lda: usize,
    a_offset: usize,
    dst: &mut [f32],
) {
    debug_assert!(dst.len() >= MR * k);

    let mut ap = a_offset;
    let mut dp = 0;

    for _ in 0..k {
        let dst_chunk = &mut dst[dp .. dp + MR];
        let mut src_idx = ap;

        for dst_val in dst_chunk[0..MR].iter_mut() {
            *dst_val = a[src_idx];
            src_idx += lda;
        }

        ap += 1;
        dp += MR;
    }
}

/// pack `mr_tail x k` micropanel for for A^T
#[inline(always)]
fn pack_at_mrxk_tail(
    k: usize,
    a: &[f32],
    lda: usize,
    a_offset: usize,
    mr_tail: usize,
    dst: &mut [f32],
) {
    debug_assert!(mr_tail > 0 && mr_tail < MR);
    debug_assert!(dst.len() >= MR * k);

    let mut ap = a_offset;
    let mut dp = 0;

    for _ in 0..k {
        let dst_chunk = &mut dst[dp .. dp + MR];
        let mut src_idx = ap;

        for dst_val in dst_chunk[0..mr_tail].iter_mut() {
            *dst_val = a[src_idx];
            src_idx += lda;
        }

        // pad 
        for dst_val in dst_chunk[mr_tail..MR].iter_mut() {
            *dst_val = 0.0;
        }

        ap += 1;
        dp += MR;
    }
}

/// pack an `mc x kc` full block of A 
#[inline(always)]
pub(crate) fn pack_a_block(
    mc: usize,
    kc: usize,
    a: &[f32],
    lda: usize,
    a_offset: usize,
    dst: &mut [f32],
) {
    let mp = mc / MR;
    let mr_tail = mc % MR;
    debug_assert!(dst.len() >= a_buf_len(mc, kc));

    let mut dp = 0;
    let mut abp = a_offset;

    // full MR-row panels
    for _ in 0..mp {
        let panel_dst = &mut dst[dp .. dp + MR * kc];
        pack_a_mrxk(kc, a, lda, abp, panel_dst);

        dp += MR * kc;
        abp += MR;
    }

    // tail rows 
    if mr_tail != 0 {
        let panel_dst = &mut dst[dp .. dp + MR * kc];
        pack_a_mrxk_tail(kc, a, lda, abp, mr_tail, panel_dst);
    }
}

/// pack an `mc x kc` full block of A^T
#[inline(always)]
pub(crate) fn pack_a_block_t(
    mc: usize,
    kc: usize,
    a: &[f32],
    lda: usize,
    a_offset: usize,
    dst: &mut [f32],
) {
    let mp = mc / MR;
    let mr_tail = mc % MR;
    debug_assert!(dst.len() >= a_buf_len(mc, kc));

    let mut dp = 0;
    let mut abp = a_offset;

    // full MR-column panels
    for _ in 0..mp {
        let panel_dst = &mut dst[dp .. dp + MR * kc];
        pack_at_mrxk(kc, a, lda, abp, panel_dst);

        dp += MR * kc;
        abp += MR * lda;
    }

    // tail columns
    if mr_tail != 0 {
        let panel_dst = &mut dst[dp .. dp + MR * kc];
        pack_at_mrxk_tail(kc, a, lda, abp, mr_tail, panel_dst);
    }
}

// B side 


/// pack one `k x NR` micropanel from B 
#[inline(always)]
fn pack_b_kxnr(
    k: usize,
    b: &[f32],
    ldb: usize,
    b_offset: usize,
    dst: &mut [f32],
) {
    debug_assert!(dst.len() >= k * NR);

    let mut dp = 0;

    for i in 0..k {
        let row_base = b_offset + i;
        let dst_row = &mut dst[dp .. dp + NR];

        for j in 0..NR {
            dst_row[j] = b[row_base + j * ldb];
        }

        dp += NR;
    }
}

/// pack `k x nr_tail` micropanel from B
#[inline(always)]
fn pack_b_kxnr_tail(
    k: usize,
    b: &[f32],
    ldb: usize,
    b_offset: usize,
    nr_tail: usize,
    dst: &mut [f32],
) {
    debug_assert!(nr_tail > 0 && nr_tail < NR);
    debug_assert!(dst.len() >= k * NR);

    let mut dp = 0;

    for i in 0..k {
        let row_base = b_offset + i;
        let dst_row = &mut dst[dp .. dp + NR];

        for j in 0..nr_tail {
            dst_row[j] = b[row_base + j * ldb];
        }
        for dst_val in dst_row[nr_tail..NR].iter_mut() {
            *dst_val = 0.0;
        }

        dp += NR;
    }
}

/// pack one `k x NR` micropanel from B^T 
#[inline(always)]
fn pack_bt_kxnr(
    k: usize,
    b: &[f32],
    ldb: usize,
    b_offset: usize,
    dst: &mut [f32],
) {
    debug_assert!(dst.len() >= k * NR);

    let mut dp = 0;

    for i in 0..k {
        let src = b_offset + i * ldb;
        let src_row = &b[src .. src + NR];
        let dst_row = &mut dst[dp .. dp + NR];

        dst_row.copy_from_slice(src_row);
        dp += NR;
    }
}

/// pack `k x nr_tail` micropanel from B 
#[inline(always)]
fn pack_bt_kxnr_tail(
    k: usize,
    b: &[f32],
    ldb: usize,
    b_offset: usize,
    nr_tail: usize,
    dst: &mut [f32],
) {
    debug_assert!(nr_tail > 0 && nr_tail < NR);
    debug_assert!(dst.len() >= k * NR);

    let mut dp = 0;

    for i in 0..k {
        let src = b_offset + i * ldb;
        let dst_row = &mut dst[dp .. dp + NR];

        // copy nr_tail
        let src_row = &b[src .. src + nr_tail];
        dst_row[..nr_tail].copy_from_slice(src_row);
        for dst_val in dst_row[nr_tail..NR].iter_mut() {
            *dst_val = 0.0;
        }

        dp += NR;
    }
}

/// pack a `kc x nc` block of B 
#[inline(always)]
pub(crate) fn pack_b_block(
    kc: usize,
    nc: usize,
    b: &[f32],
    ldb: usize,
    b_offset: usize,
    dst: &mut [f32],
) {
    let np = nc / NR;
    let nr_tail = nc % NR;
    debug_assert!(dst.len() >= b_buf_len(kc, nc));

    let mut dp = 0;
    let mut bbp = b_offset;

    // full NR-wide columns 
    for _ in 0..np {
        let panel_dst = &mut dst[dp .. dp + kc * NR];
        pack_b_kxnr(kc, b, ldb, bbp, panel_dst);

        dp += kc * NR;
        bbp += NR * ldb;
    }

    // tail NR columns
    if nr_tail != 0 {
        let panel_dst = &mut dst[dp .. dp + kc * NR];
        pack_b_kxnr_tail(kc, b, ldb, bbp, nr_tail, panel_dst);
    }
}

/// pack a `kc x nc` block of B^T 
#[inline(always)]
pub(crate) fn pack_b_block_t(
    kc: usize,
    nc: usize,
    b: &[f32],
    ldb: usize,
    b_offset: usize,
    dst: &mut [f32],
) {
    let np = nc / NR;
    let nr_tail = nc % NR;
    debug_assert!(dst.len() >= b_buf_len(kc, nc));

    let mut dp = 0;
    let mut bbp = b_offset;

    // full NR 
    for _ in 0..np {
        let panel_dst = &mut dst[dp .. dp + kc * NR];
        pack_bt_kxnr(kc, b, ldb, bbp, panel_dst);

        dp += kc * NR;
        bbp += NR; 
    }

    // tail NR 
    if nr_tail != 0 {
        let panel_dst = &mut dst[dp .. dp + kc * NR];
        pack_bt_kxnr_tail(kc, b, ldb, bbp, nr_tail, panel_dst);
    }
}
