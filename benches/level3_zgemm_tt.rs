use blas_src as _;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use cblas_sys::{cblas_zgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

use coral::enums::CoralTranspose;
use coral::level3::zgemm::zgemm;

#[inline(always)]
fn make_matrix_colmajor_c64(m: usize, n: usize, ld: usize, fill: [f64; 2]) -> Vec<f64> {
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

pub fn bench_zgemm_tt_fixed(c: &mut Criterion) {
    let n: usize = 1024;
    let (m, k) = (n, n);

    let lda = m;
    let ldb = k;
    let ldc = m;

    let alpha = [1.000123f64, 0.000789];
    let beta  = [0.000321, -0.000456];

    let a  = make_matrix_colmajor_c64(m, k, lda, [1.0, 0.0]);
    let b  = make_matrix_colmajor_c64(k, n, ldb, [1.0, 0.0]);
    let c0 = make_matrix_colmajor_c64(m, n, ldc, [2.0, 0.0]);

    c.bench_function("coral_zgemm_tt", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| {
                zgemm(
                    CoralTranspose::Transpose,
                    CoralTranspose::Transpose,
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

    c.bench_function("blas_zgemm_tt", |bch| {
        bch.iter_batched_ref(
            || c0.clone(),
            |c_buf| unsafe {
                cblas_zgemm(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasTrans),
                    black_box(CBLAS_TRANSPOSE::CblasTrans),
                    black_box(m as i32),
                    black_box(n as i32),
                    black_box(k as i32),
                    black_box(&alpha as *const [f64; 2] as *const _),
                    black_box(a.as_ptr() as *const _),
                    black_box(lda as i32),
                    black_box(b.as_ptr() as *const _),
                    black_box(ldb as i32),
                    black_box(&beta as *const [f64; 2] as *const _),
                    black_box(c_buf.as_mut_ptr() as *mut _),
                    black_box(ldc as i32),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_zgemm_tt_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();
    let mut group = c.benchmark_group("zgemm_tt");

    for &n in &sizes {
        let (m, k) = (n, n);
        let lda = m;
        let ldb = k;
        let ldc = m;

        let alpha = [1.000123, 0.000789];
        let beta  = [0.000321, -0.000456];

        let a  = make_matrix_colmajor_c64(m, k, lda, [1.0, 0.0]);
        let b  = make_matrix_colmajor_c64(k, n, ldb, [1.0, 0.0]);
        let c0 = make_matrix_colmajor_c64(m, n, ldc, [2.0, 0.0]);

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || c0.clone(),
                |c_buf| {
                    zgemm(
                        CoralTranspose::Transpose,
                        CoralTranspose::Transpose,
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
                    cblas_zgemm(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(k as i32),
                        black_box(&alpha as *const [f64; 2] as *const _),
                        black_box(a.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(b.as_ptr() as *const _),
                        black_box(ldb as i32),
                        black_box(&beta as *const [f64; 2] as *const _),
                        black_box(c_buf.as_mut_ptr() as *mut _),
                        black_box(ldc as i32),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_zgemm_tt_fixed, bench_zgemm_tt_sweep);
criterion_main!(benches);

