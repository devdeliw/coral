use crate::level3::{
    zgemm_nn::zgemm_nn,
    zgemm_nt::zgemm_nt,
    zgemm_tn::zgemm_tn,
    zgemm_tt::zgemm_tt,
    zgemm_nc::zgemm_nc,
    zgemm_tc::zgemm_tc,
    zgemm_cn::zgemm_cn,
    zgemm_ct::zgemm_ct,
    zgemm_cc::zgemm_cc,
};

use crate::enums::CoralTranspose;
use crate::level3::microkernel::c64_mrxnr::Complex64;

pub(crate) const MC: usize = 256;
pub(crate) const NC: usize = 384;
pub(crate) const KC: usize = 192;

#[inline(always)]
fn z64(x: [f64; 2]) -> Complex64 {
    Complex64 { re: x[0], im: x[1] }
}

#[inline(always)]
pub fn zgemm(
    op_a  : CoralTranspose,
    op_b  : CoralTranspose,
    m     : usize,
    n     : usize,
    k     : usize,
    alpha : [f64; 2],
    a     : *const f64,
    lda   : usize,
    b     : *const f64,
    ldb   : usize,
    beta  : [f64; 2],
    c     : *mut f64,
    ldc   : usize,
) {
    let alpha = z64(alpha);
    let beta  = z64(beta);

    match (op_a, op_b) {
        (CoralTranspose::NoTranspose,        CoralTranspose::NoTranspose)        => zgemm_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::NoTranspose,        CoralTranspose::Transpose)          => zgemm_nt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::Transpose,          CoralTranspose::NoTranspose)        => zgemm_tn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::Transpose,          CoralTranspose::Transpose)          => zgemm_tt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::NoTranspose,        CoralTranspose::ConjugateTranspose) => zgemm_nc(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::Transpose,          CoralTranspose::ConjugateTranspose) => zgemm_tc(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::NoTranspose)        => zgemm_cn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::Transpose)          => zgemm_ct(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        (CoralTranspose::ConjugateTranspose, CoralTranspose::ConjugateTranspose) => zgemm_cc(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
    }
}

