use blas_src as _;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, black_box};

use cblas_sys::cblas_sasum;

use coral_safe::types::VectorRef;
use coral_safe::level1::sasum as sasum_safe;

use coral::level1::sasum as sasum_neon; 

#[inline]
fn make_view<'a>(x: &'a [f32]) -> VectorRef<'a, f32> {
    let n = x.len();
    VectorRef::new(x, n, 1, 0).expect("x view")
}

pub fn bench_sasum_big(c: &mut Criterion) {
    let n: usize = 1_000_000;
    let x = vec![1.0f32; n];

    c.bench_function("sasum_coral_safe", |b| {
        b.iter_batched_ref(
            || make_view(&x),               
            |xv| {
                let r = sasum_safe(black_box(*xv)); 
                black_box(r);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("sasum_coral_neon", |b| {
        b.iter_batched_ref(
            || &x[..],                      
            |xs| {
                let r = sasum_neon(
                    black_box(n),
                    black_box(*xs),
                    black_box(1usize),
                );
                black_box(r);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("sasum_blas", |b| {
        b.iter_batched_ref(
            || &x[..],                      
            |xs| unsafe {
                let r = cblas_sasum(
                    black_box(n as i32),
                    black_box(xs.as_ptr()),
                    black_box(1),
                );
                black_box(r);
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_sasum_sweep(c: &mut Criterion) {
    let sizes: Vec<usize> = (128..=2_048).step_by(128).collect();
    let mut group = c.benchmark_group("sasum_sweep");

    for &n in &sizes {
        let x = vec![1.0f32; n];

        group.bench_with_input(BenchmarkId::new("coral_safe", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || make_view(&x),
                |xv| {
                    let r = sasum_safe(black_box(*xv));
                    black_box(r);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("coral_neon", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || &x[..],
                |xs| {
                    let r = sasum_neon(
                        black_box(n),
                        black_box(*xs),
                        black_box(1usize),
                    );
                    black_box(r);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("blas", n), &n, |bch, &_n| {
            bch.iter_batched_ref(
                || &x[..],
                |xs| unsafe {
                    let r = cblas_sasum(
                        black_box(n as i32),
                        black_box(xs.as_ptr()),
                        black_box(1),
                    );
                    black_box(r);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sasum_big, bench_sasum_sweep);
criterion_main!(benches);
