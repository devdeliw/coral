use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box, BenchmarkId};

use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
use coral::level2::{
    strsv, 
    dtrsv, 
    ctrsv, 
    ztrsv, 
};

use cblas_sys::{
    CBLAS_LAYOUT,
    CBLAS_TRANSPOSE,
    CBLAS_UPLO,
    CBLAS_DIAG,
    cblas_strsv,
    cblas_dtrsv,
    cblas_ctrsv,
    cblas_ztrsv,
};

pub fn bench_strsv(c: &mut Criterion) {
    let n:   usize = 1024;
    let lda: usize = n + 8;

    let matrix_upper = {
        let mut a = vec![0.0f32; lda * n];
        for j in 0..n {
            for i in 0..=j {
                a[i + j * lda] = 1.0;
            }
        }
        a
    };

    let matrix_lower = {
        let mut a = vec![0.0f32; lda * n];
        for j in 0..n {
            for i in j..n {
                a[i + j * lda] = 1.0;
            }
        }
        a
    };

    let x_init = vec![1.0f32; n];

    c.bench_function("coral_strsv_upper_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| {
                strsv(
                    black_box(CoralTriangular::UpperTriangular),
                    black_box(CoralTranspose::NoTranspose),
                    black_box(CoralDiagonal::UnitDiagonal),
                    black_box(n),
                    black_box(&matrix_upper),
                    black_box(lda),
                    black_box(x.as_mut_slice()),
                    black_box(1),
                );
                black_box(&*x);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_strsv_upper_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| unsafe {
                cblas_strsv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasUpper),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(CBLAS_DIAG::CblasUnit),
                    black_box(n as i32),
                    black_box(matrix_upper.as_ptr()),
                    black_box(lda as i32),
                    black_box(x.as_mut_ptr()),
                    black_box(1),
                );
                black_box(&*x);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_strsv_lower_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| {
                strsv(
                    black_box(CoralTriangular::LowerTriangular),
                    black_box(CoralTranspose::NoTranspose),
                    black_box(CoralDiagonal::NonUnitDiagonal),
                    black_box(n),
                    black_box(&matrix_lower),
                    black_box(lda),
                    black_box(x.as_mut_slice()),
                    black_box(1),
                );
                black_box(&*x);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_strsv_lower_notrans", |b| {
        b.iter_batched_ref(
            || x_init.clone(),
            |x| unsafe {
                cblas_strsv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_UPLO::CblasLower),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(CBLAS_DIAG::CblasNonUnit),
                    black_box(n as i32),
                    black_box(matrix_lower.as_ptr()),
                    black_box(lda as i32),
                    black_box(x.as_mut_ptr()),
                    black_box(1),
                );
                black_box(&*x);
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_strsv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group_u = c.benchmark_group("strsv_upper_notrans");
    for &n in &sizes {
        let lda = n + 8;

        let matrix_upper = {
            let mut a = vec![0.0f32; lda * n];
            for j in 0..n {
                for i in 0..=j {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f32; n];

        group_u.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    strsv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::UnitDiagonal),
                        black_box(n),
                        black_box(&matrix_upper),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_u.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_strsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasUnit),
                        black_box(n as i32),
                        black_box(matrix_upper.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_u.finish();

    let mut group_l = c.benchmark_group("strsv_lower_notrans");
    for &n in &sizes {
        let lda = n + 8;

        let matrix_lower = {
            let mut a = vec![0.0f32; lda * n];
            for j in 0..n {
                for i in j..n {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f32; n];

        group_l.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    strsv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix_lower),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_l.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_strsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix_lower.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_l.finish();

    let mut group_ut = c.benchmark_group("strsv_upper_trans");
    for &n in &sizes {
        let lda = n + 8;

        let matrix_upper = {
            let mut a = vec![0.0f32; lda * n];
            for j in 0..n {
                for i in 0..=j {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f32; n];

        group_ut.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    strsv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::UnitDiagonal),
                        black_box(n),
                        black_box(&matrix_upper),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_ut.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_strsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasUnit),
                        black_box(n as i32),
                        black_box(matrix_upper.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_ut.finish();

    let mut group_lt = c.benchmark_group("strsv_lower_trans");
    for &n in &sizes {
        let lda = n + 8;

        let matrix_lower = {
            let mut a = vec![0.0f32; lda * n];
            for j in 0..n {
                for i in j..n {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f32; n];

        group_lt.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    strsv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix_lower),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_lt.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_strsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix_lower.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_lt.finish();
}

pub fn bench_dtrsv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group_u = c.benchmark_group("dtrsv_upper_notrans");
    for &n in &sizes {
        let lda = n + 8;

        let matrix_upper = {
            let mut a = vec![0.0f64; lda * n];
            for j in 0..n {
                for i in 0..=j {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f64; n];

        group_u.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    dtrsv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::UnitDiagonal),
                        black_box(n),
                        black_box(&matrix_upper),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_u.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_dtrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasUnit),
                        black_box(n as i32),
                        black_box(matrix_upper.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_u.finish();

    let mut group_l = c.benchmark_group("dtrsv_lower_notrans");
    for &n in &sizes {
        let lda = n + 8;

        let matrix_lower = {
            let mut a = vec![0.0f64; lda * n];
            for j in 0..n {
                for i in j..n {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f64; n];

        group_l.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    dtrsv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix_lower),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_l.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_dtrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix_lower.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_l.finish();

    let mut group_ut = c.benchmark_group("dtrsv_upper_trans");
    for &n in &sizes {
        let lda = n + 8;

        let matrix_upper = {
            let mut a = vec![0.0f64; lda * n];
            for j in 0..n {
                for i in 0..=j {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f64; n];

        group_ut.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    dtrsv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::UnitDiagonal),
                        black_box(n),
                        black_box(&matrix_upper),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_ut.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_dtrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasUnit),
                        black_box(n as i32),
                        black_box(matrix_upper.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_ut.finish();

    let mut group_lt = c.benchmark_group("dtrsv_lower_trans");
    for &n in &sizes {
        let lda = n + 8;

        let matrix_lower = {
            let mut a = vec![0.0f64; lda * n];
            for j in 0..n {
                for i in j..n {
                    a[i + j * lda] = 1.0;
                }
            }
            a
        };

        let x0 = vec![1.0f64; n];

        group_lt.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    dtrsv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix_lower),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_lt.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_dtrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix_lower.as_ptr()),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_lt.finish();
}

pub fn bench_ctrsv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group_u = c.benchmark_group("ctrsv_upper_notrans");
    for &n in &sizes {
        let lda = n + 8;

        let mut matrix_upper = vec![0.0f32; 2 * lda * n];
        for j in 0..n {
            for i in 0..=j {
                let idx = 2 * (i + j * lda);
                matrix_upper[idx] = 1.0;
                matrix_upper[idx + 1] = 0.0;
            }
        }

        let mut x0 = vec![0.0f32; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group_u.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    ctrsv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::UnitDiagonal),
                        black_box(n),
                        black_box(&matrix_upper),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_u.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_ctrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasUnit),
                        black_box(n as i32),
                        black_box(matrix_upper.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_u.finish();

    let mut group_l = c.benchmark_group("ctrsv_lower_notrans");
    for &n in &sizes {
        let lda = n + 8;

        let mut matrix_lower = vec![0.0f32; 2 * lda * n];
        for j in 0..n {
            for i in j..n {
                let idx = 2 * (i + j * lda);
                matrix_lower[idx] = 1.0;
                matrix_lower[idx + 1] = 0.0;
            }
        }

        let mut x0 = vec![0.0f32; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group_l.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    ctrsv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix_lower),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_l.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_ctrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix_lower.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_l.finish();

    let mut group_ut = c.benchmark_group("ctrsv_upper_trans");
    for &n in &sizes {
        let lda = n + 8;

        let mut matrix_upper = vec![0.0f32; 2 * lda * n];
        for j in 0..n {
            for i in 0..=j {
                let idx = 2 * (i + j * lda);
                matrix_upper[idx] = 1.0;
                matrix_upper[idx + 1] = 0.0;
            }
        }

        let mut x0 = vec![0.0f32; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group_ut.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    ctrsv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::UnitDiagonal),
                        black_box(n),
                        black_box(&matrix_upper),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_ut.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_ctrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasUnit),
                        black_box(n as i32),
                        black_box(matrix_upper.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_ut.finish();

    let mut group_lt = c.benchmark_group("ctrsv_lower_trans");
    for &n in &sizes {
        let lda = n + 8;

        let mut matrix_lower = vec![0.0f32; 2 * lda * n];
        for j in 0..n {
            for i in j..n {
                let idx = 2 * (i + j * lda);
                matrix_lower[idx] = 1.0;
                matrix_lower[idx + 1] = 0.0;
            }
        }

        let mut x0 = vec![0.0f32; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group_lt.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    ctrsv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix_lower),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_lt.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_ctrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix_lower.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_lt.finish();
}

pub fn bench_ztrsv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group_u = c.benchmark_group("ztrsv_upper_notrans");
    for &n in &sizes {
        let lda = n + 8;

        let mut matrix_upper = vec![0.0f64; 2 * lda * n];
        for j in 0..n {
            for i in 0..=j {
                let idx = 2 * (i + j * lda);
                matrix_upper[idx] = 1.0;
                matrix_upper[idx + 1] = 0.0;
            }
        }

        let mut x0 = vec![0.0f64; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group_u.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    ztrsv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::UnitDiagonal),
                        black_box(n),
                        black_box(&matrix_upper),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_u.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_ztrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasUnit),
                        black_box(n as i32),
                        black_box(matrix_upper.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_u.finish();

    let mut group_l = c.benchmark_group("ztrsv_lower_notrans");
    for &n in &sizes {
        let lda = n + 8;

        let mut matrix_lower = vec![0.0f64; 2 * lda * n];
        for j in 0..n {
            for i in j..n {
                let idx = 2 * (i + j * lda);
                matrix_lower[idx] = 1.0;
                matrix_lower[idx + 1] = 0.0;
            }
        }

        let mut x0 = vec![0.0f64; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group_l.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    ztrsv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::NoTranspose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix_lower),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_l.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_ztrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix_lower.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_l.finish();

    let mut group_ut = c.benchmark_group("ztrsv_upper_trans");
    for &n in &sizes {
        let lda = n + 8;

        let mut matrix_upper = vec![0.0f64; 2 * lda * n];
        for j in 0..n {
            for i in 0..=j {
                let idx = 2 * (i + j * lda);
                matrix_upper[idx] = 1.0;
                matrix_upper[idx + 1] = 0.0;
            }
        }

        let mut x0 = vec![0.0f64; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group_ut.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    ztrsv(
                        black_box(CoralTriangular::UpperTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::UnitDiagonal),
                        black_box(n),
                        black_box(&matrix_upper),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_ut.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_ztrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasUpper),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasUnit),
                        black_box(n as i32),
                        black_box(matrix_upper.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_ut.finish();

    let mut group_lt = c.benchmark_group("ztrsv_lower_trans");
    for &n in &sizes {
        let lda = n + 8;

        let mut matrix_lower = vec![0.0f64; 2 * lda * n];
        for j in 0..n {
            for i in j..n {
                let idx = 2 * (i + j * lda);
                matrix_lower[idx] = 1.0;
                matrix_lower[idx + 1] = 0.0;
            }
        }

        let mut x0 = vec![0.0f64; 2 * n];
        for i in 0..n {
            x0[2 * i] = 1.0;
            x0[2 * i + 1] = -1.0;
        }

        group_lt.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| {
                    ztrsv(
                        black_box(CoralTriangular::LowerTriangular),
                        black_box(CoralTranspose::Transpose),
                        black_box(CoralDiagonal::NonUnitDiagonal),
                        black_box(n),
                        black_box(&matrix_lower),
                        black_box(lda),
                        black_box(x.as_mut_slice()),
                        black_box(1usize),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });

        group_lt.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || x0.clone(),
                |x| unsafe {
                    cblas_ztrsv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_UPLO::CblasLower),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(CBLAS_DIAG::CblasNonUnit),
                        black_box(n as i32),
                        black_box(matrix_lower.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                    black_box(&*x);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_lt.finish();
}

criterion_group!(
    benches,
    bench_strsv,
    bench_strsv_n,
    bench_dtrsv_n,
    bench_ctrsv_n,
    bench_ztrsv_n
);
criterion_main!(benches);

