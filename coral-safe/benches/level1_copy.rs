use blas_src as _;
use criterion::{
    criterion_group,
    criterion_main,
    BatchSize,
    BenchmarkId,
    Criterion,
    black_box
};
use cblas_sys::cblas_scopy;

use coral_safe::types::{VectorRef, VectorMut};
use coral_safe::level1::scopy as scopy_safe;
use coral::level1::scopy as scopy_neon;

#[inline]
fn make_views<'a>(
    x: &'a [f32],
    y: &'a mut [f32],
) -> (VectorRef<'a, f32>, VectorMut<'a, f32>) {
    let n  = x.len();
    let xv = VectorRef::new(x, n, 1, 0).expect("x view");
    let yv = VectorMut::new(y, n, 1, 0).expect("y view");
    (xv, yv)
}

pub fn bench_scopy_big(c: &mut Criterion) {
    let n: usize = 1_000_000;

    let x = vec![1.0f32; n];

    c.bench_function("scopy_coral_safe", |b| {
        let mut y = vec![0.0f32; n];

        b.iter_batched_ref(
            || (),
            |_| {
                y.fill(0.0);
                let (xv, yv) = make_views(&x, &mut y);
                black_box(scopy_safe(black_box(xv), black_box(yv)));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("scopy_coral_neon", |b| {
        let mut y = vec![0.0f32; n];

        b.iter_batched_ref(
            || (),
            |_| {
                y.fill(0.0);
                black_box(scopy_neon(
                    black_box(n),
                    black_box(&x),
                    black_box(1usize),
                    black_box(&mut y[..]),
                    black_box(1usize),
                ));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("scopy_blas", |b| {
        let mut y = vec![0.0f32; n];

        b.iter_batched_ref(
            || (),
            |_| unsafe {
                y.fill(0.0);
                black_box(cblas_scopy(
                    black_box(n as i32),
                    black_box(x.as_ptr()),
                    black_box(1),
                    black_box(y.as_mut_ptr()),
                    black_box(1),
                ));
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_scopy_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("scopy_sweep");

    for &n in &sizes {
        let x = vec![1.0f32; n];

        group.bench_with_input(BenchmarkId::new("coral_safe", n), &n, |bch, &_n| {
            let mut y = vec![0.0f32; n];

            bch.iter_batched_ref(
                || (),
                |_| {
                    y.fill(0.0);
                    let (xv, yv) = make_views(&x, &mut y);
                    black_box(scopy_safe(black_box(xv), black_box(yv)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("coral_neon", n), &n, |bch, &_n| {
            let mut y = vec![0.0f32; n];

            bch.iter_batched_ref(
                || (),
                |_| {
                    y.fill(0.0);
                    black_box(scopy_neon(
                        black_box(n),
                        black_box(&x),
                        black_box(1usize),
                        black_box(&mut y[..]),
                        black_box(1usize),
                    ));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            let mut y = vec![0.0f32; n];

            bch.iter_batched_ref(
                || (),
                |_| unsafe {
                    y.fill(0.0);
                    black_box(cblas_scopy(
                        black_box(n as i32),
                        black_box(x.as_ptr()),
                        black_box(1),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                    ));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_scopy_big, bench_scopy_sweep);
criterion_main!(benches);

