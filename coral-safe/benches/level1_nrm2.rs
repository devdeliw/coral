use blas_src as _;
use criterion::{
    criterion_group,
    criterion_main,
    BatchSize,
    BenchmarkId,
    Criterion,
    black_box
};
use cblas_sys::cblas_snrm2;
use coral_safe::types::VectorRef;
use coral_safe::level1::snrm2 as snrm2_safe;
use coral::level1::snrm2 as snrm2_neon;

#[inline]
fn make_view<'a>(x: &'a [f32]) -> VectorRef<'a, f32> {
    let n  = x.len();
    VectorRef::new(x, n, 1, 0).expect("x view")
}

pub fn bench_snrm2_big(c: &mut Criterion) {
    let n: usize = 1000_000;

    let x = vec![1.0f32; n];

    c.bench_function("snrm2_coral_safe", |b| {
        let xv = make_view(&x);
        b.iter_batched_ref(
            || (),
            |_| {
                black_box(snrm2_safe(black_box(xv)));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("snrm2_coral_neon", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                black_box(snrm2_neon(
                    black_box(n),
                    black_box(&x),
                    black_box(1usize),
                ));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("snrm2_blas", |b| {
        b.iter_batched_ref(
            || (),
            |_| unsafe {
                black_box(cblas_snrm2(
                    black_box(n as i32),
                    black_box(x.as_ptr()),
                    black_box(1),
                ));
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_snrm2_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();

    let mut group = c.benchmark_group("snrm2_sweep");

    for &n in &sizes {
        let x = vec![1.0f32; n];

        group.bench_with_input(BenchmarkId::new("coral_safe", n), &n, |bch, &_n| {
            let xv = make_view(&x);
            bch.iter_batched_ref(
                || (),
                |_| {
                    black_box(snrm2_safe(black_box(xv)));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("coral_neon", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || (),
                |_| {
                    black_box(snrm2_neon(
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
                    black_box(cblas_snrm2(
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

criterion_group!(benches, bench_snrm2_big, bench_snrm2_sweep);
criterion_main!(benches);

