use rand::Rng;

use blas_src as _;
use criterion::{
    criterion_group,
    criterion_main,
    BatchSize,
    BenchmarkId,
    Criterion,
    black_box,
};
use cblas_sys::cblas_isamax;

use coral_safe::types::VectorRef;
use coral_safe::level1::isamax as isamax_safe;
use coral::level1::isamax as isamax_neon;

#[inline]
fn make_view<'a>(x: &'a [f32]) -> VectorRef<'a, f32> {
    let n = x.len();
    VectorRef::new(x, n, 1, 0).expect("x view")
}

pub fn bench_isamax_big(c: &mut Criterion) {
    let n: usize = 1000000;

    let mut rng = rand::thread_rng();
    let x: Vec<f32> = (0..n)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect();

    c.bench_function("isamax_coral_safe", |b| {
        let xv = make_view(&x);
        b.iter_batched_ref(
            || (),
            |_| {
                black_box(isamax_safe(black_box(xv)));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("isamax_coral_neon", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                black_box(isamax_neon(
                    black_box(n),
                    black_box(&x),
                    black_box(1usize),
                ));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("isamax_blas", |b| {
        b.iter_batched_ref(
            || (),
            |_| unsafe {
                black_box(cblas_isamax(
                    black_box(n as i32),
                    black_box(x.as_ptr()),
                    black_box(1),
                ));
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_isamax_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("isamax_sweep");

    for &n in &sizes {
        let x = vec![1.0f32; n];

        group.bench_with_input(BenchmarkId::new("coral_safe", n), &n, |bch, &_n| {
            let xv = make_view(&x);
            bch.iter_batched_ref(
                || (),
                |_| {
                    black_box(isamax_safe(black_box(xv)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("coral_neon", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || (),
                |_| {
                    black_box(isamax_neon(
                        black_box(n),
                        black_box(&x),
                        black_box(1usize),
                    ));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || (),
                |_| unsafe {
                    black_box(cblas_isamax(
                        black_box(n as i32),
                        black_box(x.as_ptr()),
                        black_box(1),
                    ));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_isamax_big, bench_isamax_sweep);
criterion_main!(benches);

