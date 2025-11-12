use blas_src as _;

use criterion::{
    criterion_group,
    criterion_main,
    BatchSize,
    BenchmarkId,
    Criterion,
    black_box,
};
use cblas_sys::cblas_sswap;

use coral_safe::types::VectorMut;
use coral_safe::level1::sswap as sswap_safe;
use coral::level1::sswap as sswap_neon;

#[inline]
fn make_views<'a>(
    x: &'a mut [f32],
    y: &'a mut [f32],
) -> (VectorMut<'a, f32>, VectorMut<'a, f32>) {
    let n  = x.len();
    let xv = VectorMut::new(x, n, 1, 0).expect("x view");
    let yv = VectorMut::new(y, n, 1, 0).expect("y view");
    (xv, yv)
}

pub fn bench_sswap_big(c: &mut Criterion) {
    let n: usize = 1_000_000;

    c.bench_function("sswap_coral_safe", |b| {
        let mut x = vec![1.0f32; n];
        let mut y = vec![0.0f32; n];

        b.iter_batched_ref(
            || (),
            |_| {
                x.fill(1.0);
                y.fill(0.0);
                let (xv, yv) = make_views(&mut x, &mut y);
                black_box(sswap_safe(black_box(xv), black_box(yv)));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("sswap_coral_neon", |b| {
        let mut x = vec![1.0f32; n];
        let mut y = vec![0.0f32; n];

        b.iter_batched_ref(
            || (),
            |_| {
                x.fill(1.0);
                y.fill(0.0);
                black_box(sswap_neon(
                    black_box(n),
                    black_box(&mut x[..]),
                    black_box(1usize),
                    black_box(&mut y[..]),
                    black_box(1usize),
                ));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("sswap_blas", |b| {
        let mut x = vec![1.0f32; n];
        let mut y = vec![0.0f32; n];

        b.iter_batched_ref(
            || (),
            |_| unsafe {
                x.fill(1.0);
                y.fill(0.0);
                black_box(cblas_sswap(
                    black_box(n as i32),
                    black_box(x.as_mut_ptr()),
                    black_box(1),
                    black_box(y.as_mut_ptr()),
                    black_box(1),
                ));
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_sswap_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("sswap_sweep");

    for &n in &sizes {
        group.bench_with_input(BenchmarkId::new("coral_safe", n), &n, |bch, &_n| {
            let mut x = vec![1.0f32; n];
            let mut y = vec![0.0f32; n];

            bch.iter_batched_ref(
                || (),
                |_| {
                    x.fill(1.0);
                    y.fill(0.0);
                    let (xv, yv) = make_views(&mut x, &mut y);
                    black_box(sswap_safe(black_box(xv), black_box(yv)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("coral_neon", n), &n, |bch, &_n| {
            let mut x = vec![1.0f32; n];
            let mut y = vec![0.0f32; n];

            bch.iter_batched_ref(
                || (),
                |_| {
                    x.fill(1.0);
                    y.fill(0.0);
                    black_box(sswap_neon(
                        black_box(n),
                        black_box(&mut x[..]),
                        black_box(1usize),
                        black_box(&mut y[..]),
                        black_box(1usize),
                    ));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            let mut x = vec![1.0f32; n];
            let mut y = vec![0.0f32; n];

            bch.iter_batched_ref(
                || (),
                |_| unsafe {
                    x.fill(1.0);
                    y.fill(0.0);
                    black_box(cblas_sswap(
                        black_box(n as i32),
                        black_box(x.as_mut_ptr()),
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

criterion_group!(benches, bench_sswap_big, bench_sswap_sweep);
criterion_main!(benches);
