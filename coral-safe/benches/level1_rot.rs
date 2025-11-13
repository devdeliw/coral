use blas_src as _;
use criterion::{
    criterion_group,
    criterion_main,
    BatchSize,
    BenchmarkId,
    Criterion,
    black_box,
};
use cblas_sys::cblas_srot;

use coral_safe::types::VectorMut;
use coral_safe::level1::srot as srot_safe;
use coral::level1::srot as srot_neon;

#[inline]
fn make_views_mut<'a>(
    x: &'a mut [f32],
    y: &'a mut [f32],
) -> (VectorMut<'a, f32>, VectorMut<'a, f32>) {
    let n = x.len();
    let xv = VectorMut::new(x, n, 1, 0).expect("x view");
    let yv = VectorMut::new(y, n, 1, 0).expect("y view");
    (xv, yv)
}

pub fn bench_srot_big(c: &mut Criterion) {
    let n: usize = 1000000;
    let cval: f32 = 0.8;
    let sval: f32 = 0.6;

    let mut x = vec![1.0f32; n];
    let mut y = vec![0.5f32; n];

    c.bench_function("srot_coral_safe", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                let (xv, yv) = make_views_mut(&mut x, &mut y);
                black_box(srot_safe(xv, yv, black_box(cval), black_box(sval)));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("srot_coral_neon", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                black_box(srot_neon(
                    black_box(n),
                    black_box(&mut x),
                    black_box(1usize),
                    black_box(&mut y),
                    black_box(1usize),
                    black_box(cval),
                    black_box(sval),
                ));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("srot_blas", |b| {
        b.iter_batched_ref(
            || (),
            |_| unsafe {
                black_box(cblas_srot(
                    black_box(n as i32),
                    black_box(x.as_mut_ptr()),
                    black_box(1),
                    black_box(y.as_mut_ptr()),
                    black_box(1),
                    black_box(cval),
                    black_box(sval),
                ));
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_srot_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("srot_sweep");

    for &n in &sizes {
        let mut x = vec![1.0f32; n];
        let mut y = vec![0.5f32; n];
        let cval: f32 = 0.8;
        let sval: f32 = 0.6;

        group.bench_with_input(BenchmarkId::new("coral_safe", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || (),
                |_| {
                    let (xv, yv) = make_views_mut(&mut x, &mut y);
                    black_box(srot_safe(xv, yv, black_box(cval), black_box(sval)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("coral_neon", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || (),
                |_| {
                    black_box(srot_neon(
                        black_box(n),
                        black_box(&mut x),
                        black_box(1usize),
                        black_box(&mut y),
                        black_box(1usize),
                        black_box(cval),
                        black_box(sval),
                    ));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || (),
                |_| unsafe {
                    black_box(cblas_srot(
                        black_box(n as i32),
                        black_box(x.as_mut_ptr()),
                        black_box(1),
                        black_box(y.as_mut_ptr()),
                        black_box(1),
                        black_box(cval),
                        black_box(sval),
                    ));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_srot_big, bench_srot_sweep);
criterion_main!(benches);
