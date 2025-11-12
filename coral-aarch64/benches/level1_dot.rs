use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, BenchmarkId, black_box};

use coral::level1::{
    sdot,
    ddot,
    cdotu,
    cdotc,
    zdotu,
    zdotc,
};

use cblas_sys::{
    cblas_sdot,
    cblas_ddot,
    cblas_cdotu_sub,
    cblas_cdotc_sub,
    cblas_zdotu_sub,
    cblas_zdotc_sub,
};

pub fn bench_dot(c: &mut Criterion) {
    let n:   usize = 1_000_000;
    let inc: i32   = 1;

    let xr_f32 = vec![1.0f32; n];
    let yr_f32 = vec![1.0f32; n];
    let xr_f64 = vec![1.0f64; n];
    let yr_f64 = vec![1.0f64; n];

    let mut xc_f32 = vec![0.0f32; 2 * n];
    let mut yc_f32 = vec![0.0f32; 2 * n];
    let mut xc_f64 = vec![0.0f64; 2 * n];
    let mut yc_f64 = vec![0.0f64; 2 * n];

    for i in 0..n {
        xc_f32[2 * i]     =  1.0;
        xc_f32[2 * i + 1] = -1.0;
        yc_f32[2 * i]     =  1.0;
        yc_f32[2 * i + 1] =  1.0;

        xc_f64[2 * i]     =  1.0;
        xc_f64[2 * i + 1] = -1.0;
        yc_f64[2 * i]     =  1.0;
        yc_f64[2 * i + 1] =  1.0;
    }

    c.bench_function("coral_sdot", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = sdot(
                    black_box(n),
                    black_box(&xr_f32),
                    black_box(1usize),
                    black_box(&yr_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_sdot", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_sdot(
                        black_box(n as i32),
                        black_box(xr_f32.as_ptr()),
                        black_box(inc),
                        black_box(yr_f32.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_ddot", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = ddot(
                    black_box(n),
                    black_box(&xr_f64),
                    black_box(1usize),
                    black_box(&yr_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_ddot", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = unsafe {
                    cblas_ddot(
                        black_box(n as i32),
                        black_box(xr_f64.as_ptr()),
                        black_box(inc),
                        black_box(yr_f64.as_ptr()),
                        black_box(inc),
                    )
                };
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_cdotu", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = cdotu(
                    black_box(n),
                    black_box(&xc_f32),
                    black_box(1usize),
                    black_box(&yc_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_cdotu", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut out = [0.0f32; 2];
                unsafe {
                    cblas_cdotu_sub(
                        black_box(n as i32),
                        black_box(xc_f32.as_ptr() as *const _),
                        black_box(inc),
                        black_box(yc_f32.as_ptr() as *const _),
                        black_box(inc),
                        black_box(out.as_mut_ptr() as *mut _),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_cdotc", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = cdotc(
                    black_box(n),
                    black_box(&xc_f32),
                    black_box(1usize),
                    black_box(&yc_f32),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_cdotc", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut out = [0.0f32; 2];
                unsafe {
                    cblas_cdotc_sub(
                        black_box(n as i32),
                        black_box(xc_f32.as_ptr() as *const _),
                        black_box(inc),
                        black_box(yc_f32.as_ptr() as *const _),
                        black_box(inc),
                        black_box(out.as_mut_ptr() as *mut _),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zdotu", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = zdotu(
                    black_box(n),
                    black_box(&xc_f64),
                    black_box(1usize),
                    black_box(&yc_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zdotu", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut out = [0.0f64; 2];
                unsafe {
                    cblas_zdotu_sub(
                        black_box(n as i32),
                        black_box(xc_f64.as_ptr() as *const _),
                        black_box(inc),
                        black_box(yc_f64.as_ptr() as *const _),
                        black_box(inc),
                        black_box(out.as_mut_ptr() as *mut _),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("coral_zdotc", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = zdotc(
                    black_box(n),
                    black_box(&xc_f64),
                    black_box(1usize),
                    black_box(&yc_f64),
                    black_box(1usize),
                );
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("blas_zdotc", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut out = [0.0f64; 2];
                unsafe {
                    cblas_zdotc_sub(
                        black_box(n as i32),
                        black_box(xc_f64.as_ptr() as *const _),
                        black_box(inc),
                        black_box(yc_f64.as_ptr() as *const _),
                        black_box(inc),
                        black_box(out.as_mut_ptr() as *mut _),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_sdot_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("sdot");

    for &n in &sizes {
        let x = vec![1.0f32; n];
        let y = vec![1.0f32; n];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let _ = sdot(
                        black_box(n),
                        black_box(&x),
                        black_box(1usize),
                        black_box(&y),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let _ = unsafe {
                        cblas_sdot(
                            black_box(n as i32),
                            black_box(x.as_ptr()),
                            black_box(1),
                            black_box(y.as_ptr()),
                            black_box(1),
                        )
                    };
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_ddot_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("ddot");

    for &n in &sizes {
        let x = vec![1.0f64; n];
        let y = vec![1.0f64; n];

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let _ = ddot(
                        black_box(n),
                        black_box(&x),
                        black_box(1usize),
                        black_box(&y),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let _ = unsafe {
                        cblas_ddot(
                            black_box(n as i32),
                            black_box(x.as_ptr()),
                            black_box(1),
                            black_box(y.as_ptr()),
                            black_box(1),
                        )
                    };
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_cdotu_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("cdotu");

    for &n in &sizes {
        let mut x = vec![0.0f32; 2 * n];
        let mut y = vec![0.0f32; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
            y[2 * i] = 1.0;
            y[2 * i + 1] = 1.0;
        }

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let _ = cdotu(
                        black_box(n),
                        black_box(&x),
                        black_box(1usize),
                        black_box(&y),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let mut out = [0.0f32; 2];
                    unsafe {
                        cblas_cdotu_sub(
                            black_box(n as i32),
                            black_box(x.as_ptr() as *const _),
                            black_box(1),
                            black_box(y.as_ptr() as *const _),
                            black_box(1),
                            black_box(out.as_mut_ptr() as *mut _),
                        );
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_cdotc_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("cdotc");

    for &n in &sizes {
        let mut x = vec![0.0f32; 2 * n];
        let mut y = vec![0.0f32; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
            y[2 * i] = 1.0;
            y[2 * i + 1] = 1.0;
        }

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let _ = cdotc(
                        black_box(n),
                        black_box(&x),
                        black_box(1usize),
                        black_box(&y),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let mut out = [0.0f32; 2];
                    unsafe {
                        cblas_cdotc_sub(
                            black_box(n as i32),
                            black_box(x.as_ptr() as *const _),
                            black_box(1),
                            black_box(y.as_ptr() as *const _),
                            black_box(1),
                            black_box(out.as_mut_ptr() as *mut _),
                        );
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_zdotu_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("zdotu");

    for &n in &sizes {
        let mut x = vec![0.0f64; 2 * n];
        let mut y = vec![0.0f64; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
            y[2 * i] = 1.0;
            y[2 * i + 1] = 1.0;
        }

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let _ = zdotu(
                        black_box(n),
                        black_box(&x),
                        black_box(1usize),
                        black_box(&y),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let mut out = [0.0f64; 2];
                    unsafe {
                        cblas_zdotu_sub(
                            black_box(n as i32),
                            black_box(x.as_ptr() as *const _),
                            black_box(1),
                            black_box(y.as_ptr() as *const _),
                            black_box(1),
                            black_box(out.as_mut_ptr() as *mut _),
                        );
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

pub fn bench_zdotc_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("zdotc");

    for &n in &sizes {
        let mut x = vec![0.0f64; 2 * n];
        let mut y = vec![0.0f64; 2 * n];
        for i in 0..n {
            x[2 * i] = 1.0;
            x[2 * i + 1] = -1.0;
            y[2 * i] = 1.0;
            y[2 * i + 1] = 1.0;
        }

        group.bench_with_input(BenchmarkId::new("coral", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let _ = zdotc(
                        black_box(n),
                        black_box(&x),
                        black_box(1usize),
                        black_box(&y),
                        black_box(1usize),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched(
                || (),
                |_| {
                    let mut out = [0.0f64; 2];
                    unsafe {
                        cblas_zdotc_sub(
                            black_box(n as i32),
                            black_box(x.as_ptr() as *const _),
                            black_box(1),
                            black_box(y.as_ptr() as *const _),
                            black_box(1),
                            black_box(out.as_mut_ptr() as *mut _),
                        );
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dot,
    bench_sdot_sweep,
    bench_ddot_sweep,
    bench_cdotu_sweep,
    bench_cdotc_sweep,
    bench_zdotu_sweep,
    bench_zdotc_sweep
);
criterion_main!(benches);

