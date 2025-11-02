use blas_src as _;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use cblas_sys::{cblas_cgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

use coral::enums::CoralTranspose;
use coral::level3::cgemm;

use faer::{mat, Parallelism};
use faer::complex_native::c32;
use faer::linalg::matmul::matmul as faer_cgemm;

#[inline(always)]
fn make_matrix_colmajor_c32(
    m   : usize, 
    n   : usize, 
    ld  : usize, 
    fill: [f32; 2]
) -> Vec<f32> {
    assert!(ld >= m);
    let mut a = vec![0.0; 2 * ld * n];
    for j in 0..n {
        for i in 0..m {
            let idx = 2 * (i + j * ld);
            a[idx + 0] = fill[0];
            a[idx + 1] = fill[1];
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
) -> faer::MatRef<'a, c32> {
    unsafe { 
        mat::from_raw_parts::<c32>(ptr as *const c32, m, n, 1, ld as isize)
    }
}
#[inline(always)]
fn faer_mut<'a>(
    ptr: *mut f32,
    m  : usize,
    n  : usize, 
    ld : usize
) -> faer::MatMut<'a, c32> {
    unsafe { 
        mat::from_raw_parts_mut::<c32>(ptr as *mut c32, m, n, 1, ld as isize)
    }
}

pub fn bench_cgemm_nn_fixed(c: &mut Criterion) {
    let n: usize = 1024;
    let (m, k) = (n, n);

    let lda = m;
    let ldb = k;
    let ldc = m;

    let alpha = [1.000_123, 0.000_789];
    let beta  = [0.000_321, -0.000_456];

    let a  = make_matrix_colmajor_c32(m, k, lda, [1.0, 0.0]);
    let b  = make_matrix_colmajor_c32(k, n, ldb, [1.0, 0.0]);
    let c0 = make_matrix_colmajor_c32(m, n, ldc, [2.0, 0.0]);

    c.bench_function("coral_cgemm_nn", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| {
                cgemm(
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

    c.bench_function("blas_cgemm_nn", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| unsafe {
                cblas_cgemm(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(m as i32),
                    black_box(n as i32),
                    black_box(k as i32),
                    black_box(&alpha as *const [f32; 2]),
                    black_box(a.as_ptr() as *const [f32; 2]),
                    black_box(lda as i32),
                    black_box(b.as_ptr() as *const [f32; 2]),
                    black_box(ldb as i32),
                    black_box(&beta as *const [f32; 2]),
                    black_box(c_buf.as_mut_ptr() as *mut [f32; 2]),
                    black_box(ldc as i32),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("faer_cgemm_nn_fixed", |bch| {
        bch.iter_batched(
            || {
                let c_buf  = c0.clone();
                let a_ref  = faer_ref(a.as_ptr(), m, k, lda);
                let b_ref  = faer_ref(b.as_ptr(), k, n, ldb);
                let alpha_c = c32::new(alpha[0], alpha[1]);
                let beta_c  = c32::new(beta[0],  beta[1]);
                (c_buf, a_ref, b_ref, alpha_c, beta_c)
            },
            |(mut c_buf, a_ref, b_ref, alpha_c, beta_c)| {
                let mut c_mut = faer_mut(c_buf.as_mut_ptr(), m, n, ldc);

                faer_cgemm(
                    c_mut.as_mut(),
                    a_ref,
                    b_ref,
                    Some(black_box(beta_c)),
                    black_box(alpha_c),
                    Parallelism::None,
                );

                black_box(c_buf);
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_cgemm_nn_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();
    let mut group = c.benchmark_group("cgemm_nn");

    for &n in &sizes {
        let (m, k) = (n, n);
        let lda = m;
        let ldb = k;
        let ldc = m;

        let alpha = [1.000_123, 0.000_789];
        let beta  = [0.000_321, -0.000_456];

        let a  = make_matrix_colmajor_c32(m, k, lda, [1.0, 0.0]);
        let b  = make_matrix_colmajor_c32(k, n, ldb, [1.0, 0.0]);
        let c0 = make_matrix_colmajor_c32(m, n, ldc, [2.0, 0.0]);

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || c0.clone(),
                |c_buf| {
                    cgemm(
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

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || c0.clone(),
                |c_buf| unsafe {
                    cblas_cgemm(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(k as i32),
                        black_box(&alpha as *const [f32; 2]),
                        black_box(a.as_ptr() as *const [f32; 2]),
                        black_box(lda as i32),
                        black_box(b.as_ptr() as *const [f32; 2]),
                        black_box(ldb as i32),
                        black_box(&beta as *const [f32; 2]),
                        black_box(c_buf.as_mut_ptr() as *mut [f32; 2]),
                        black_box(ldc as i32),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        c.bench_function("faer_cgemm_nn_fixed", |bch| {
            bch.iter_batched(
                || {
                    let c_buf  = c0.clone();
                    let a_ref  = faer_ref(a.as_ptr(), m, k, lda);
                    let b_ref  = faer_ref(b.as_ptr(), k, n, ldb);
                    let alpha_c = c32::new(alpha[0], alpha[1]);
                    let beta_c  = c32::new(beta[0],  beta[1]);
                    (c_buf, a_ref, b_ref, alpha_c, beta_c)
                },
                |(mut c_buf, a_ref, b_ref, alpha_c, beta_c)| {
                    let mut c_mut = faer_mut(c_buf.as_mut_ptr(), m, n, ldc);

                    faer_cgemm(
                        c_mut.as_mut(),
                        a_ref,
                        b_ref,
                        Some(black_box(beta_c)),
                        black_box(alpha_c),
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

criterion_group!(benches, bench_cgemm_nn_fixed, bench_cgemm_nn_sweep);
criterion_main!(benches);

