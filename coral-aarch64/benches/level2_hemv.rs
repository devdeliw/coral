use blas_src as _;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion,
};
use coral::enums::CoralTriangular;
use coral::level2::chemv;
use coral::level2::zhemv;

use cblas_sys::{
    cblas_chemv, cblas_zhemv, CBLAS_LAYOUT, CBLAS_UPLO,
};

fn bench_chemv(c: &mut Criterion) {
    let n: usize  = 1024;
    let lda: usize = n;

    let alpha: [f32; 2] = [1.000_123, -0.999_877];
    let beta:  [f32; 2] = [0.000_321,  0.000_123];

    let mut matrix = vec![0.0f32; 2 * lda * n];
    for j in 0..n {
        for i in 0..n {
            let idx = 2 * (i + j * lda);
            matrix[idx] = 1.0;
            matrix[idx + 1] = 0.0;
        }
    }

    let mut x = vec![0.0f32; 2 * n];
    let mut y0 = vec![0.0f32; 2 * n];
    for i in 0..n {
        x[2 * i] = 1.0;
        x[2 * i + 1] = -1.0;
        y0[2 * i] = 2.0;
        y0[2 * i + 1] = 0.0;
    }

    c.bench_function("coral_chemv_upper", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                chemv(
                    black_box(CoralTriangular::UpperTriangular),
                    black_box(n),
                    black_box(alpha),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(&x),
                    black_box(1usize),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_chemv_upper", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_chemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasUpper),
                    black_box(n as i32),
                    black_box(alpha.as_ptr() as *const _),
                    black_box(matrix.as_ptr() as *const _),
                    black_box(lda as i32),
                    black_box(x.as_ptr() as *const _),
                    black_box(1),
                    black_box(beta.as_ptr() as *const _),
                    black_box(y.as_mut_ptr() as *mut _),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_chemv_lower", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                chemv(
                    black_box(CoralTriangular::LowerTriangular),
                    black_box(n),
                    black_box(alpha),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(&x),
                    black_box(1usize),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_chemv_lower", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_chemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasLower),
                    black_box(n as i32),
                    black_box(alpha.as_ptr() as *const _),
                    black_box(matrix.as_ptr() as *const _),
                    black_box(lda as i32),
                    black_box(x.as_ptr() as *const _),
                    black_box(1),
                    black_box(beta.as_ptr() as *const _),
                    black_box(y.as_mut_ptr() as *mut _),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_chemv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group_u = c.benchmark_group("chemv_upper");
    for &n in &sizes {
        let lda = n;
        let alpha: [f32; 2] = [1.000_123, -0.999_877];
        let beta:  [f32; 2] = [0.000_321,  0.000_123];

        let mut matrix = vec![0.0f32; 2 * lda * n];
        for j in 0..n {
            for i in 0..n {
                let idx = 2 * (i + j * lda);
                matrix[idx] = 1.0;
                matrix[idx + 1] = 0.0;
            }
        }

        let mut x = vec![0.0f32; 2 * n];
        let mut y0 = vec![0.0f32; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
            y0[2 * i] = 2.0;
            y0[2 * i + 1] = 0.0;
        }

        group_u.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    chemv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x),
                        black_box(1usize),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_u.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_chemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(matrix.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_ptr() as *const _),
                        black_box(1),
                        black_box(beta.as_ptr() as *const _),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_u.finish();

    let mut group_l = c.benchmark_group("chemv_lower");
    for &n in &sizes {
        let lda = n;
        let alpha: [f32; 2] = [1.000_123, -0.999_877];
        let beta:  [f32; 2] = [0.000_321,  0.000_123];

        let mut matrix = vec![0.0f32; 2 * lda * n];
        for j in 0..n {
            for i in 0..n {
                let idx = 2 * (i + j * lda);
                matrix[idx] = 1.0;
                matrix[idx + 1] = 0.0;
            }
        }

        let mut x = vec![0.0f32; 2 * n];
        let mut y0 = vec![0.0f32; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
            y0[2 * i] = 2.0;
            y0[2 * i + 1] = 0.0;
        }

        group_l.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    chemv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x),
                        black_box(1usize),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_l.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_chemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(matrix.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_ptr() as *const _),
                        black_box(1),
                        black_box(beta.as_ptr() as *const _),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_l.finish();
}

fn bench_zhemv(c: &mut Criterion) {
    let n: usize  = 1024;
    let lda: usize = n;

    let alpha: [f64; 2] = [1.000_123_f64, -0.999_877_f64];
    let beta:  [f64; 2] = [0.000_321_f64,  0.000_123_f64];

    let mut matrix = vec![0.0f64; 2 * lda * n];
    for j in 0..n {
        for i in 0..n {
            let idx = 2 * (i + j * lda);
            matrix[idx] = 1.0;
            matrix[idx + 1] = 0.0;
        }
    }

    let mut x = vec![0.0f64; 2 * n];
    let mut y0 = vec![0.0f64; 2 * n];
    for i in 0..n {
        x[2 * i] = 1.0;
        x[2 * i + 1] = -1.0;
        y0[2 * i] = 2.0;
        y0[2 * i + 1] = 0.0;
    }

    c.bench_function("coral_zhemv_upper", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                zhemv(
                    black_box(CoralTriangular::UpperTriangular),
                    black_box(n),
                    black_box(alpha),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(&x),
                    black_box(1usize),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zhemv_upper", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_zhemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasUpper),
                    black_box(n as i32),
                    black_box(alpha.as_ptr() as *const _),
                    black_box(matrix.as_ptr() as *const _),
                    black_box(lda as i32),
                    black_box(x.as_ptr() as *const _),
                    black_box(1),
                    black_box(beta.as_ptr() as *const _),
                    black_box(y.as_mut_ptr() as *mut _),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zhemv_lower", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                zhemv(
                    black_box(CoralTriangular::LowerTriangular),
                    black_box(n),
                    black_box(alpha),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(&x),
                    black_box(1usize),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zhemv_lower", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_zhemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasLower),
                    black_box(n as i32),
                    black_box(alpha.as_ptr() as *const _),
                    black_box(matrix.as_ptr() as *const _),
                    black_box(lda as i32),
                    black_box(x.as_ptr() as *const _),
                    black_box(1),
                    black_box(beta.as_ptr() as *const _),
                    black_box(y.as_mut_ptr() as *mut _),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_zhemv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group_u = c.benchmark_group("zhemv_upper");
    for &n in &sizes {
        let lda = n;
        let alpha: [f64; 2] = [1.000_123_f64, -0.999_877_f64];
        let beta:  [f64; 2] = [0.000_321_f64,  0.000_123_f64];

        let mut matrix = vec![0.0f64; 2 * lda * n];
        for j in 0..n {
            for i in 0..n {
                let idx = 2 * (i + j * lda);
                matrix[idx] = 1.0;
                matrix[idx + 1] = 0.0;
            }
        }

        let mut x = vec![0.0f64; 2 * n];
        let mut y0 = vec![0.0f64; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
            y0[2 * i] = 2.0;
            y0[2 * i + 1] = 0.0;
        }

        group_u.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    zhemv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x),
                        black_box(1usize),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_u.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_zhemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(matrix.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_ptr() as *const _),
                        black_box(1),
                        black_box(beta.as_ptr() as *const _),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_u.finish();

    let mut group_l = c.benchmark_group("zhemv_lower");
    for &n in &sizes {
        let lda = n;
        let alpha: [f64; 2] = [1.000_123_f64, -0.999_877_f64];
        let beta:  [f64; 2] = [0.000_321_f64,  0.000_123_f64];

        let mut matrix = vec![0.0f64; 2 * lda * n];
        for j in 0..n {
            for i in 0..n {
                let idx = 2 * (i + j * lda);
                matrix[idx] = 1.0;
                matrix[idx + 1] = 0.0;
            }
        }

        let mut x = vec![0.0f64; 2 * n];
        let mut y0 = vec![0.0f64; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
            y0[2 * i] = 2.0;
            y0[2 * i + 1] = 0.0;
        }

        group_l.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    zhemv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x),
                        black_box(1usize),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_l.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_zhemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(matrix.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_ptr() as *const _),
                        black_box(1),
                        black_box(beta.as_ptr() as *const _),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_l.finish();
}

criterion_group!(benches, bench_chemv, bench_chemv_n, bench_zhemv, bench_zhemv_n);
criterion_main!(benches);

