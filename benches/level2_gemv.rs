use blas_src as _; 
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, black_box, BenchmarkId};

use coral::enums::CoralTranspose; 
use coral::level2::{
    sgemv, 
    dgemv, 
    cgemv, 
    zgemv, 
};

use cblas_sys::{
    CBLAS_LAYOUT, 
    CBLAS_TRANSPOSE, 
    cblas_sgemv, 
    cblas_dgemv,
    cblas_cgemv,
    cblas_zgemv,
}; 

pub fn bench_sgemv(c: &mut Criterion) { 
    let m:   usize = 1024; 
    let n:   usize = 1024; 
    let lda: usize = m; 

    let alpha: f32 = 1.000123; 
    let beta : f32 = 0.000321; 
    let matrix     = vec![1.0f32; lda * n]; 

    let x_notrans  = vec![1.0f32; n]; 
    let x_trans    = vec![1.0f32; m];  

    c.bench_function("coral_sgemv_notrans", |b| { 
        let y0 = vec![2.0f32; m]; 
        b.iter_batched_ref(
            || y0.clone(), 
            |y| { 
                sgemv(
                    black_box(CoralTranspose::NoTranspose), 
                    black_box(m), 
                    black_box(n), 
                    black_box(alpha), 
                    black_box(&matrix), 
                    black_box(lda), 
                    black_box(&x_notrans), 
                    black_box(1),
                    black_box(beta), 
                    black_box(y.as_mut_slice()), 
                    black_box(1),
                );
            },
            BatchSize::SmallInput, 
        );
    });

    c.bench_function("blas_sgemv_notrans", |b| {
        let y0 = vec![2.0f32; m];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_sgemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                    black_box(m as i32), 
                    black_box(n as i32),
                    black_box(alpha),
                    black_box(matrix.as_ptr()),
                    black_box(lda as i32),
                    black_box(x_notrans.as_ptr()), 
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_ptr()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    }); 

    c.bench_function("coral_sgemv_trans", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                sgemv(
                    black_box(CoralTranspose::Transpose),
                    black_box(m),
                    black_box(n), 
                    black_box(alpha),
                    black_box(&matrix), 
                    black_box(lda),
                    black_box(&x_trans),
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_sgemv_trans", |b| {
        let y0 = vec![2.0f32; n];
        b.iter_batched_ref(
            || y0.clone(),
            |y| unsafe {
                cblas_sgemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasTrans),
                    black_box(m as i32), 
                    black_box(n as i32),
                    black_box(alpha),
                    black_box(matrix.as_ptr()),
                    black_box(lda as i32),
                    black_box(x_trans.as_ptr()),
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_ptr()),
                    black_box(1),
                );
            },
            BatchSize::SmallInput,
        );
    });
}


pub fn bench_sgemv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group = c.benchmark_group("sgemv_notrans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: f32 = 1.000123;
        let beta:  f32 = 0.000321;

        let matrix    = vec![1.0f32; lda * n];
        let x_notrans = vec![1.0f32; n];
        let y0        = vec![2.0f32; m];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    sgemv(
                        black_box(CoralTranspose::NoTranspose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_notrans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_sgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x_notrans.as_ptr()),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();

    let mut group_t = c.benchmark_group("sgemv_trans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: f32 = 1.000123;
        let beta:  f32 = 0.000321;

        let matrix = vec![1.0f32; lda * n];
        let x_trans = vec![1.0f32; m];
        let y0      = vec![2.0f32; n];

        group_t.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    sgemv(
                        black_box(CoralTranspose::Transpose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_trans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_t.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_sgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x_trans.as_ptr()),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_t.finish();
}

pub fn bench_dgemv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group = c.benchmark_group("dgemv_notrans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: f64 = 1.000123_f64;
        let beta:  f64 = 0.000321_f64;

        let matrix    = vec![1.0f64; lda * n];
        let x_notrans = vec![1.0f64; n];
        let y0        = vec![2.0f64; m];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    dgemv(
                        black_box(CoralTranspose::NoTranspose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_notrans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_dgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x_notrans.as_ptr()),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();

    let mut group_t = c.benchmark_group("dgemv_trans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: f64 = 1.000123_f64;
        let beta:  f64 = 0.000321_f64;

        let matrix = vec![1.0f64; lda * n];
        let x_trans = vec![1.0f64; m];
        let y0      = vec![2.0f64; n];

        group_t.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    dgemv(
                        black_box(CoralTranspose::Transpose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_trans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_t.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_dgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(matrix.as_ptr()),
                        black_box(lda as i32),
                        black_box(x_trans.as_ptr()),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group_t.finish();
}

pub fn bench_cgemv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group = c.benchmark_group("cgemv_notrans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: [f32; 2] = [1.000123, 0.000321];
        let beta:  [f32; 2] = [0.000321, 0.0];

        let matrix    = vec![1.0f32; 2 * lda * n];
        let mut x_notrans = vec![0.0f32; 2 * n];
        for i in 0..n { x_notrans[2*i] = 1.0; x_notrans[2*i+1] = -1.0; }
        let y0        = vec![2.0f32; 2 * m];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    cgemv(
                        black_box(CoralTranspose::NoTranspose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_notrans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_cgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(matrix.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x_notrans.as_ptr() as *const _),
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
    group.finish();

    let mut group_t = c.benchmark_group("cgemv_trans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: [f32; 2] = [1.000123, 0.000321];
        let beta:  [f32; 2] = [0.000321, 0.0];

        let matrix = vec![1.0f32; 2 * lda * n];
        let mut x_trans = vec![0.0f32; 2 * m];
        for i in 0..m { x_trans[2*i] = 1.0; x_trans[2*i+1] = -1.0; }
        let y0      = vec![2.0f32; 2 * n];

        group_t.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    cgemv(
                        black_box(CoralTranspose::Transpose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_trans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_t.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_cgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(matrix.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x_trans.as_ptr() as *const _),
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
    group_t.finish();

    let mut group_h = c.benchmark_group("cgemv_conjtrans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: [f32; 2] = [1.000123, 0.000321];
        let beta:  [f32; 2] = [0.000321, 0.0];

        let matrix = vec![1.0f32; 2 * lda * n];
        let mut x_h = vec![0.0f32; 2 * m];
        for i in 0..m { x_h[2*i] = 1.0; x_h[2*i+1] = -1.0; }
        let y0     = vec![2.0f32; 2 * n];

        group_h.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(|| y0.clone(), |y| {
                cgemv(
                    black_box(CoralTranspose::ConjugateTranspose),
                    black_box(m),
                    black_box(n),
                    black_box(alpha),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(&x_h),
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1),
                );
            }, BatchSize::SmallInput);
        });

        group_h.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(|| y0.clone(), |y| unsafe {
                cblas_cgemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasConjTrans),
                    black_box(m as i32),
                    black_box(n as i32),
                    black_box(alpha.as_ptr() as *const _),
                    black_box(matrix.as_ptr() as *const _),
                    black_box(lda as i32),
                    black_box(x_h.as_ptr() as *const _),
                    black_box(1),
                    black_box(beta.as_ptr() as *const _),
                    black_box(y.as_mut_ptr() as *mut _),
                    black_box(1),
                );
            }, BatchSize::SmallInput);
        });
    }
    group_h.finish();
}

pub fn bench_zgemv_n(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2048).step_by(128).collect();

    let mut group = c.benchmark_group("zgemv_notrans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: [f64; 2] = [1.000123_f64, 0.000321_f64];
        let beta:  [f64; 2] = [0.000321_f64, 0.0];

        let matrix    = vec![1.0f64; 2 * lda * n];
        let mut x_notrans = vec![0.0f64; 2 * n];
        for i in 0..n { x_notrans[2*i] = 1.0; x_notrans[2*i+1] = -1.0; }
        let y0        = vec![2.0f64; 2 * m];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    zgemv(
                        black_box(CoralTranspose::NoTranspose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_notrans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_zgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasNoTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(matrix.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x_notrans.as_ptr() as *const _),
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
    group.finish();

    let mut group_t = c.benchmark_group("zgemv_trans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: [f64; 2] = [1.000123_f64, 0.000321_f64];
        let beta:  [f64; 2] = [0.000321_f64, 0.0];

        let matrix = vec![1.0f64; 2 * lda * n];
        let mut x_trans = vec![0.0f64; 2 * m];
        for i in 0..m { x_trans[2*i] = 1.0; x_trans[2*i+1] = -1.0; }
        let y0      = vec![2.0f64; 2 * n];

        group_t.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| {
                    zgemv(
                        black_box(CoralTranspose::Transpose),
                        black_box(m),
                        black_box(n),
                        black_box(alpha),
                        black_box(&matrix),
                        black_box(lda),
                        black_box(&x_trans),
                        black_box(1),
                        black_box(beta),
                        black_box(y.as_mut_slice()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group_t.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_zgemv(
                        black_box(CBLAS_LAYOUT::CblasColMajor),
                        black_box(CBLAS_TRANSPOSE::CblasTrans),
                        black_box(m as i32),
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(matrix.as_ptr() as *const _),
                        black_box(lda as i32),
                        black_box(x_trans.as_ptr() as *const _),
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
    group_t.finish();

    let mut group_h = c.benchmark_group("zgemv_conjtrans");
    for &n in &sizes {
        let m = n;
        let lda = m;

        let alpha: [f64; 2] = [1.000123_f64, 0.000321_f64];
        let beta:  [f64; 2] = [0.000321_f64, 0.0];

        let matrix = vec![1.0f64; 2 * lda * n];
        let mut x_h = vec![0.0f64; 2 * m];
        for i in 0..m { x_h[2*i] = 1.0; x_h[2*i+1] = -1.0; }
        let y0     = vec![2.0f64; 2 * n];

        group_h.bench_with_input(BenchmarkId::new("coral", n), &n, |b, &_n| {
            b.iter_batched_ref(|| y0.clone(), |y| {
                zgemv(
                    black_box(CoralTranspose::ConjugateTranspose),
                    black_box(m),
                    black_box(n),
                    black_box(alpha),
                    black_box(&matrix),
                    black_box(lda),
                    black_box(&x_h),
                    black_box(1),
                    black_box(beta),
                    black_box(y.as_mut_slice()),
                    black_box(1),
                );
            }, BatchSize::SmallInput);
        });

        group_h.bench_with_input(BenchmarkId::new("blas", n), &n, |b, &_n| {
            b.iter_batched_ref(|| y0.clone(), |y| unsafe {
                cblas_zgemv(
                    black_box(CBLAS_LAYOUT::CblasColMajor),
                    black_box(CBLAS_TRANSPOSE::CblasConjTrans),
                    black_box(m as i32),
                    black_box(n as i32),
                    black_box(alpha.as_ptr() as *const _),
                    black_box(matrix.as_ptr() as *const _),
                    black_box(lda as i32),
                    black_box(x_h.as_ptr() as *const _),
                    black_box(1),
                    black_box(beta.as_ptr() as *const _),
                    black_box(y.as_mut_ptr() as *mut _),
                    black_box(1),
                );
            }, BatchSize::SmallInput);
        });
    }
    group_h.finish();
}


criterion_group!( 
    benches, 
    bench_sgemv,
    bench_sgemv_n,
    bench_dgemv_n,
    bench_cgemv_n, 
    bench_zgemv_n 
);
criterion_main!(benches);
