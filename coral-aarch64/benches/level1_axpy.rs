use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, BenchmarkId, black_box};

use coral::level1::{
    saxpy,
    daxpy,
    caxpy,
    zaxpy,
};

use cblas_sys::{
    cblas_saxpy,
    cblas_daxpy,
    cblas_caxpy,
    cblas_zaxpy,
};

pub fn bench_axpy(c: &mut Criterion) {
    let n:   usize = 1_000_000;

    let alpha_f32: f32 = 1.000123;
    let alpha_f64: f64 = 1.000123;
    let alpha_c32: [f32; 2] = [1.000123, -0.999877];
    let alpha_c64: [f64; 2] = [1.000123, -0.999877];

    let x_f32  = vec![1.0f32; n];
    let x_f64  = vec![1.0f64; n];
    let y0_f32 = vec![0.0f32; n];
    let y0_f64 = vec![0.0f64; n];

    let mut x_c32  = vec![0.0f32; 2 * n];
    let     y0_c32 = vec![0.0f32; 2 * n];
    let mut x_c64  = vec![0.0f64; 2 * n];
    let     y0_c64 = vec![0.0f64; 2 * n];

    for i in 0..n {
        x_c32[2 * i]     =  1.0;
        x_c32[2 * i + 1] = -1.0;
        x_c64[2 * i]     =  1.0;
        x_c64[2 * i + 1] = -1.0;
    }

    c.bench_function("coral_saxpy", |b| {
        b.iter_batched_ref(
            || y0_f32.clone(),
            |y| {
                saxpy(
                    black_box(n),
                    black_box(alpha_f32),
                    black_box(&x_f32),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_saxpy", |b| {
        b.iter_batched_ref(
            || y0_f32.clone(),
            |y| {
                unsafe {
                    cblas_saxpy(
                        black_box(n as i32),
                        black_box(alpha_f32),
                        black_box(x_f32.as_ptr()),
                        black_box(1),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_daxpy", |b| {
        b.iter_batched_ref(
            || y0_f64.clone(),
            |y| {
                daxpy(
                    black_box(n),
                    black_box(alpha_f64),
                    black_box(&x_f64),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_daxpy", |b| {
        b.iter_batched_ref(
            || y0_f64.clone(),
            |y| {
                unsafe {
                    cblas_daxpy(
                        black_box(n as i32),
                        black_box(alpha_f64),
                        black_box(x_f64.as_ptr()),
                        black_box(1),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_caxpy", |b| {
        b.iter_batched_ref(
            || y0_c32.clone(),
            |y| {
                caxpy(
                    black_box(n),
                    black_box(alpha_c32),
                    black_box(&x_c32),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_caxpy", |b| {
        b.iter_batched_ref(
            || y0_c32.clone(),
            |y| {
                unsafe {
                    cblas_caxpy(
                        black_box(n as i32),
                        black_box(alpha_c32.as_ptr() as *const _),
                        black_box(x_c32.as_ptr() as *const _),
                        black_box(1),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zaxpy", |b| {
        b.iter_batched_ref(
            || y0_c64.clone(),
            |y| {
                zaxpy(
                    black_box(n),
                    black_box(alpha_c64),
                    black_box(&x_c64),
                    black_box(1usize),
                    black_box(y.as_mut_slice()),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zaxpy", |b| {
        b.iter_batched_ref(
            || y0_c64.clone(),
            |y| {
                unsafe {
                    cblas_zaxpy(
                        black_box(n as i32),
                        black_box(alpha_c64.as_ptr() as *const _),
                        black_box(x_c64.as_ptr() as *const _),
                        black_box(1),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_saxpy_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let alpha: f32 = 1.000123;

    let mut group = c.benchmark_group("saxpy");

    for &n in &sizes {
        let x = vec![1.0f32; n];
        let y0 = vec![0.0f32; n];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| {
                    saxpy(
                        black_box(n),
                        black_box(alpha),
                        black_box(&x),
                        black_box(1usize),
                        black_box(y.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_saxpy(
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(x.as_ptr()),
                        black_box(1),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_daxpy_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let alpha: f64 = 1.000123;

    let mut group = c.benchmark_group("daxpy");

    for &n in &sizes {
        let x = vec![1.0f64; n];
        let y0 = vec![0.0f64; n];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| {
                    daxpy(
                        black_box(n),
                        black_box(alpha),
                        black_box(&x),
                        black_box(1usize),
                        black_box(y.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_daxpy(
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(x.as_ptr()),
                        black_box(1),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_caxpy_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let alpha: [f32; 2] = [1.000123, -0.999877];

    let mut group = c.benchmark_group("caxpy");

    for &n in &sizes {
        let mut x = vec![0.0f32; 2 * n];
        let y0 = vec![0.0f32; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
        }

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| {
                    caxpy(
                        black_box(n),
                        black_box(alpha),
                        black_box(&x),
                        black_box(1usize),
                        black_box(y.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_caxpy(
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(x.as_ptr() as *const _),
                        black_box(1),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_zaxpy_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let alpha: [f64; 2] = [1.000123, -0.999877];

    let mut group = c.benchmark_group("zaxpy");

    for &n in &sizes {
        let mut x = vec![0.0f64; 2 * n];
        let y0 = vec![0.0f64; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
        }

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| {
                    zaxpy(
                        black_box(n),
                        black_box(alpha),
                        black_box(&x),
                        black_box(1usize),
                        black_box(y.as_mut_slice()),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| unsafe {
                    cblas_zaxpy(
                        black_box(n as i32),
                        black_box(alpha.as_ptr() as *const _),
                        black_box(x.as_ptr() as *const _),
                        black_box(1),
                        black_box(y.as_mut_ptr() as *mut _),
                        black_box(1),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_axpy, bench_saxpy_sweep, bench_daxpy_sweep, bench_caxpy_sweep, bench_zaxpy_sweep);
criterion_main!(benches);

