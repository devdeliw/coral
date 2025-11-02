
use blas_src as _; 
use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use coral::enums::CoralTranspose;
use coral::level3::sgemm;

use faer::{mat, Parallelism};
use faer::linalg::matmul::matmul as faer_sgemm;


#[inline(always)]
fn make_matrix_colmajor(
    m   : usize,
    n   : usize,
    ld  : usize,
    fill: f32
) -> Vec<f32> {
    assert!(ld >= m);
    let mut a = vec![0.0f32; ld * n];
    for j in 0..n {
        for i in 0..m {
            a[i + j * ld] = fill;
        }
    }
    a
}

#[inline(always)]
fn faer_ref<'a>(
    ptr: *const f32, 
    m  : usize,
    n  : usize,
    ld : usize
) -> faer::MatRef<'a, f32> {
    unsafe {
        mat::from_raw_parts::<f32>(ptr, m, n, 1, ld as isize)
    } 
}
#[inline(always)]
fn faer_mut<'a>(
    ptr: *mut f32,
    m  : usize, 
    n  : usize, 
    ld : usize
) -> faer::MatMut<'a, f32> {
    unsafe {
        mat::from_raw_parts_mut::<f32>(ptr, m, n, 1, ld as isize)
    } 
}

pub fn bench_sgemm_nn_fixed(c: &mut Criterion) {
    let n: usize = 1024;
    let (m, k) = (n, n);

    let lda = m;
    let ldb = k;
    let ldc = m;

    let alpha: f32 = 1.000_123;
    let beta:  f32 = 0.000_321;

    let a  = make_matrix_colmajor(m, k, lda, 1.0);
    let b  = make_matrix_colmajor(k, n, ldb, 1.0);
    let c0 = make_matrix_colmajor(m, n, ldc, 2.0);

    // coral
    c.bench_function("coral_sgemm_nn_fixed", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| {
                sgemm(
                    CoralTranspose::NoTranspose,
                    CoralTranspose::NoTranspose,
                    black_box(m),
                    black_box(n),
                    black_box(k),
                    black_box(alpha),
                    black_box(a.as_ptr()),
                    black_box(lda),
                    black_box(b.as_ptr()),
                    black_box(ldb),
                    black_box(beta),
                    black_box(c_buf.as_mut_ptr()),
                    black_box(ldc),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_sgemm_nn_fixed", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| unsafe {
                cblas_sgemm(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(m as i32),
                    black_box(n as i32),
                    black_box(k as i32),
                    black_box(alpha),
                    black_box(a.as_ptr()),
                    black_box(lda as i32),
                    black_box(b.as_ptr()),
                    black_box(ldb as i32),
                    black_box(beta),
                    black_box(c_buf.as_mut_ptr()),
                    black_box(ldc as i32),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("faer_sgemm_nn_fixed", |bch| {
        bch.iter_batched(
            || {
                let c_buf = c0.clone();
                let a_ref = faer_ref(a.as_ptr(), m, k, lda);
                let b_ref = faer_ref(b.as_ptr(), k, n, ldb);
                (c_buf, a_ref, b_ref)
            },
            |(mut c_buf, a_ref, b_ref)| {
                let mut c_mut = faer_mut(c_buf.as_mut_ptr(), m, n, ldc);
                faer_sgemm(
                    c_mut.as_mut(),
                    a_ref,
                    b_ref,
                    Some(black_box(beta)),
                    black_box(alpha),
                    Parallelism::None,
                );
                black_box(c_buf);
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_sgemm_nn_sweep(c: &mut Criterion) {

    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();
    let mut group = c.benchmark_group("sgemm_nn");

    for &n in &sizes {
        let (m, k) = (n, n);
        let lda = m;
        let ldb = k;
        let ldc = m;

        let alpha: f32 = 1.000_123;
        let beta:  f32 = 0.000_321;

        let a  = make_matrix_colmajor(m, k, lda, 1.0);
        let b  = make_matrix_colmajor(k, n, ldb, 1.0);
        let c0 = make_matrix_colmajor(m, n, ldc, 2.0);

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched(
                || c0.clone(),
                |mut c_buf| {
                    sgemm(
                        CoralTranspose::NoTranspose,
                        CoralTranspose::NoTranspose,
                        black_box(m),
                        black_box(n),
                        black_box(k),
                        black_box(alpha),
                        black_box(a.as_ptr()),
                        black_box(lda),
                        black_box(b.as_ptr()),
                        black_box(ldb),
                        black_box(beta),
                        black_box(c_buf.as_mut_ptr()),
                        black_box(ldc),
                    );
                    black_box(c_buf);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("cblas", n), &n, |bch, &_n| {
            bch.iter_batched(
                || c0.clone(),
                |mut c_buf| {
                    unsafe {
                        cblas_sgemm(
                            black_box(CBLAS_LAYOUT::CblasColMajor),
                            black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                            black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                            black_box(m as i32),
                            black_box(n as i32),
                            black_box(k as i32),
                            black_box(alpha),
                            black_box(a.as_ptr()),
                            black_box(lda as i32),
                            black_box(b.as_ptr()),
                            black_box(ldb as i32),
                            black_box(beta),
                            black_box(c_buf.as_mut_ptr()),
                            black_box(ldc as i32),
                        );
                    }
                    black_box(c_buf);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("faer", n), &n, |bch, &_n| {
            bch.iter_batched(
                || {
                    let c_buf = c0.clone();
                    let a_ref = faer_ref(a.as_ptr(), m, k, lda);
                    let b_ref = faer_ref(b.as_ptr(), k, n, ldb);
                    (c_buf, a_ref, b_ref)
                },
                |(mut c_buf, a_ref, b_ref)| {
                    let mut c_mut = faer_mut(c_buf.as_mut_ptr(), m, n, ldc);
                    faer_sgemm(
                        c_mut.as_mut(),
                        a_ref,
                        b_ref,
                        Some(black_box(beta)),
                        black_box(alpha),
                        Parallelism::None, 
                    );
                    black_box(c_buf);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}


criterion_group!(benches, bench_sgemm_nn_fixed, bench_sgemm_nn_sweep);
criterion_main!(benches);

