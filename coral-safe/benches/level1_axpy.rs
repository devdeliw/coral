use blas_src as _; 
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, black_box};

use cblas_sys::cblas_saxpy;

use coral_safe::types::{VectorMut, VectorRef};
use coral_safe::level1::saxpy as saxpy_safe;

use coral::level1::saxpy as saxpy_neon;

#[inline]
fn make_views<'a>(
    x: &'a [f32],
    y: &'a mut [f32],
) -> (VectorRef<'a, f32>, VectorMut<'a, f32>) {
    let n = x.len();
    let xv = VectorRef::new(x, n, 1, 0).expect("x view");
    let yv = VectorMut::new(y, n, 1, 0).expect("y view");
    (xv, yv)
}

pub fn bench_saxpy_big(c: &mut Criterion) {
    let n: usize = 1000_000;
    let alpha: f32 = 1.000123;

    let x = vec![1.0f32; n];
    let y0 = vec![0.0f32; n];

    c.bench_function("saxpy_coral_safe", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                let (xv, yv) = make_views(&x, y.as_mut_slice());
                saxpy_safe(black_box(alpha), black_box(xv), black_box(yv));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("saxpy_coral_neon", |b| {
        b.iter_batched_ref(
            || y0.clone(),
            |y| {
                saxpy_neon(
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

    c.bench_function("saxpy_blas", |b| {
        b.iter_batched_ref(
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

pub fn bench_saxpy_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let alpha: f32 = 1.000123;

    let mut group = c.benchmark_group("saxpy_sweep");

    for &n in &sizes {
        let x = vec![1.0f32; n];
        let y0 = vec![0.0f32; n];

        group.bench_with_input(BenchmarkId::new("coral_safe", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| {
                    let (xv, yv) = make_views(&x, y.as_mut_slice());
                    saxpy_safe(black_box(alpha), black_box(xv), black_box(yv));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("coral_neon", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || y0.clone(),
                |y| {
                    saxpy_neon(
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

criterion_group!(benches, bench_saxpy_big, bench_saxpy_sweep);
criterion_main!(benches);

