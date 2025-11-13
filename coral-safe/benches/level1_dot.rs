use blas_src as _;
use criterion::{
    criterion_group,
    criterion_main,
    BatchSize,
    BenchmarkId,
    Criterion,
    black_box,
};
use cblas_sys::cblas_sdot;

use coral_safe::types::VectorRef;
use coral_safe::level1::sdot as sdot_safe;
use coral::level1::sdot as sdot_neon;

#[inline]
fn make_view<'a>(x: &'a [f32]) -> VectorRef<'a, f32> {
    let n = x.len();
    VectorRef::new(x, n, 1, 0).expect("x view")
}

pub fn bench_sdot_big(c: &mut Criterion) {
    let n: usize = 1000_000;

    let x = vec![1.0f32; n];
    let y = vec![0.5f32; n];

    let xv = make_view(&x);
    let yv = make_view(&y);

    c.bench_function("sdot_coral_safe", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                let res = sdot_safe(black_box(xv), black_box(yv));
                black_box(res);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("sdot_coral_neon", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                let res = sdot_neon(
                    black_box(n),
                    black_box(&x),
                    black_box(1usize),
                    black_box(&y),
                    black_box(1usize),
                );
                black_box(res);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("sdot_blas", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                let res = unsafe {
                    cblas_sdot(
                        black_box(n as i32),
                        black_box(x.as_ptr()),
                        black_box(1),
                        black_box(y.as_ptr()),
                        black_box(1),
                    )
                };
                black_box(res);
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_sdot_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("sdot_sweep");

    for &n in &sizes {
        let x = vec![1.0f32; n];
        let y = vec![0.5f32; n];

        let xv = make_view(&x);
        let yv = make_view(&y);

        group.bench_with_input(BenchmarkId::new("coral_safe", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || (),
                |_| {
                    let res = sdot_safe(black_box(xv), black_box(yv));
                    black_box(res);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("coral_neon", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || (),
                |_| {
                    let res = sdot_neon(
                        black_box(n),
                        black_box(&x),
                        black_box(1usize),
                        black_box(&y),
                        black_box(1usize),
                    );
                    black_box(res);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || (),
                |_| {
                    let res = unsafe {
                        cblas_sdot(
                            black_box(n as i32),
                            black_box(x.as_ptr()),
                            black_box(1),
                            black_box(y.as_ptr()),
                            black_box(1),
                        )
                    };
                    black_box(res);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sdot_big, bench_sdot_sweep);
criterion_main!(benches);

