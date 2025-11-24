use super::common::{
    make_strided_mat,
    assert_close,
    CoralResult,
    RTOL,
    ATOL,
};

use blas_src as _;
use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use coral_safe::types::{MatrixRef, MatrixMut, CoralTranspose};
use coral_safe::level3::sgemm;

#[inline(always)]
fn to_cblas(op: CoralTranspose) -> CBLAS_TRANSPOSE {
    match op {
        CoralTranspose::NoTrans => CBLAS_TRANSPOSE::CblasNoTrans,
        CoralTranspose::Trans   => CBLAS_TRANSPOSE::CblasTrans,
    }
}

#[inline(always)]
fn cblas_sgemm_ref(
    op_a: CoralTranspose,
    op_b: CoralTranspose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasColMajor,
            to_cblas(op_a),
            to_cblas(op_b),
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }
}

fn run_case(
    op_a: CoralTranspose,
    op_b: CoralTranspose,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
) -> CoralResult {
    let (a_rows, a_cols) = match op_a {
        CoralTranspose::NoTrans => (m, k), 
        CoralTranspose::Trans   => (k, m), 
    };
    let (b_rows, b_cols) = match op_b {
        CoralTranspose::NoTrans => (k, n), 
        CoralTranspose::Trans   => (n, k), 
    };

    assert!(lda >= a_rows && ldb >= b_rows && ldc >= m);

    let abuf = make_strided_mat(a_rows, a_cols, lda);
    let bbuf = make_strided_mat(b_rows, b_cols, ldb);
    let c_init = make_strided_mat(m, n, ldc);

    let mut c_coral = c_init.clone();

    let acoral = MatrixRef::new(&abuf, a_rows, a_cols, lda, 0)?;
    let bcoral = MatrixRef::new(&bbuf, b_rows, b_cols, ldb, 0)?;
    let ccoral = MatrixMut::new(&mut c_coral, m, n, ldc, 0)?;

    sgemm(op_a, op_b, alpha, beta, acoral, bcoral, ccoral);

    let mut c_ref = c_init.clone();
    cblas_sgemm_ref(
        op_a,
        op_b,
        m as i32,
        n as i32,
        k as i32,
        alpha,
        abuf.as_ptr(),
        lda as i32,
        bbuf.as_ptr(),
        ldb as i32,
        beta,
        c_ref.as_mut_ptr(),
        ldc as i32,
    );

    assert_close(&c_coral, &c_ref, RTOL, ATOL);
    Ok(())
}

fn run_all_ops(
    m: usize,
    n: usize,
    k: usize,
    lda_n: usize,
    lda_t: usize,
    ldb_n: usize,
    ldb_t: usize,
    ldc: usize,
) -> CoralResult {
    let cases = &[
        (1.0f32, 0.0f32),
        (0.5, 1.0),
        (0.75, -0.25),
        (0.0, 0.7),
    ];

    let ops = &[
        //(CoralTranspose::NoTrans, CoralTranspose::NoTrans, lda_n, ldb_n),
        (CoralTranspose::NoTrans, CoralTranspose::Trans,   lda_n, ldb_t),
        (CoralTranspose::Trans,   CoralTranspose::NoTrans, lda_t, ldb_n),
        (CoralTranspose::Trans,   CoralTranspose::Trans,   lda_t, ldb_t),
    ];

    for &(alpha, beta) in cases {
        for &(op_a, op_b, lda, ldb) in ops {
            run_case(op_a, op_b, m, n, k, lda, ldb, ldc, alpha, beta)?;
        }
    }

    Ok(())
}

#[test]
fn small_all_ops() -> CoralResult {
    let (m, n, k) = (5, 7, 4);
    let lda_n = m;
    let lda_t = k;
    let ldb_n = k;
    let ldb_t = n;
    let ldc   = m;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc)
}

#[test]
fn not_block_multiple_all_ops() -> CoralResult {
    let (m, n, k) = (13, 14, 11);
    let lda_n = m;
    let lda_t = k;
    let ldb_n = k;
    let ldb_t = n;
    let ldc   = m;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc)
}

#[test]
fn padded_lds_all_ops() -> CoralResult {
    let (m, n, k) = (17, 19, 13);
    let lda_n = m + 3;
    let lda_t = k + 2;
    let ldb_n = k + 5;
    let ldb_t = n + 4;
    let ldc   = m + 7;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc)
}

#[test]
fn medium_square_all_ops() -> CoralResult {
    let (m, n, k) = (128, 128, 128);
    let lda_n = m;
    let lda_t = k;
    let ldb_n = k;
    let ldb_t = n;
    let ldc   = m;
    run_all_ops(m, n, k, lda_n, lda_t, ldb_n, ldb_t, ldc)
}

#[test]
fn rectangular_all_ops() -> CoralResult {
    run_all_ops(64, 192, 15, 64, 15, 15, 192, 64)?;
    run_all_ops(192, 64, 15, 192, 15, 15, 64, 192)?;
    Ok(())
}

