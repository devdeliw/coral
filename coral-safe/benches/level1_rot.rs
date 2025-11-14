use blas_src as _;
use criterion::{
    criterion_group,
    criterion_main,
    BatchSize,
    BenchmarkId,
    Criterion,
    black_box,
};
use cblas_sys::{
    cblas_srot,
    cblas_srotg,
    cblas_srotm,
    cblas_srotmg,
};

use coral_safe::types::VectorMut;
use coral_safe::level1::{
    srot   as srot_safe,
    srotg  as srotg_safe,
    srotm  as srotm_safe,
    srotmg as srotmg_safe,
};

use coral::level1::{
    srot   as srot_neon,
    srotg  as srotg_neon,
    srotm  as srotm_neon,
    srotmg as srotmg_neon,
};

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

pub fn bench_srotm_big(c: &mut Criterion) {
    let n: usize = 1000000;

    // Full 2x2 modified Givens matrix (flag = -1.0)
    let param: [f32; 5] = [
        -1.0f32, // flag
        0.9f32,  // h11
        -0.3f32, // h21
        0.4f32,  // h12
        1.1f32,  // h22
    ];

    let mut x = vec![1.0f32; n];
    let mut y = vec![0.5f32; n];

    c.bench_function("srotm_coral_safe", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                let (xv, yv) = make_views_mut(&mut x, &mut y);
                black_box(srotm_safe(
                    xv,
                    yv,
                    black_box(&param),
                ));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("srotm_coral_neon", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                black_box(srotm_neon(
                    black_box(n),
                    black_box(&mut x),
                    black_box(1usize),
                    black_box(&mut y),
                    black_box(1usize),
                    black_box(&param),
                ));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("srotm_blas", |b| {
        b.iter_batched_ref(
            || (),
            |_| unsafe {
                black_box(cblas_srotm(
                    black_box(n as i32),
                    black_box(x.as_mut_ptr()),
                    black_box(1),
                    black_box(y.as_mut_ptr()),
                    black_box(1),
                    black_box(param.as_ptr()),
                ));
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_srotmg_big(c: &mut Criterion) {
    let d1_0 = 1.5f32;
    let d2_0 = 2.0f32;
    let x1_0 = -0.75f32;
    let y1   = 0.5f32;

    c.bench_function("srotmg_coral_safe", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                let mut d1    = black_box(d1_0);
                let mut d2    = black_box(d2_0);
                let mut x1    = black_box(x1_0);
                let mut param = [0.0f32; 5];

                srotmg_safe(
                    &mut d1,
                    &mut d2,
                    &mut x1,
                    black_box(y1),
                    &mut param,
                );

                black_box((d1, d2, x1, param));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("srotmg_coral_neon", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                let mut d1    = black_box(d1_0);
                let mut d2    = black_box(d2_0);
                let mut x1    = black_box(x1_0);
                let mut param = [0.0f32; 5];

                srotmg_neon(
                    &mut d1,
                    &mut d2,
                    &mut x1,
                    black_box(y1),
                    &mut param,
                );

                black_box((d1, d2, x1, param));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("srotmg_blas", |b| {
        b.iter_batched_ref(
            || (),
            |_| unsafe {
                let mut d1    = black_box(d1_0);
                let mut d2    = black_box(d2_0);
                let mut x1    = black_box(x1_0);
                let mut param = [0.0f32; 5];

                black_box(cblas_srotmg(
                    black_box(&mut d1 as *mut f32),
                    black_box(&mut d2 as *mut f32),
                    black_box(&mut x1 as *mut f32),
                    black_box(y1),
                    black_box(param.as_mut_ptr()),
                ));

                black_box((d1, d2, x1, param));
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_srotg_big(c: &mut Criterion) {
    let a0 = 0.5f32;
    let b0 = -1.25f32;

    c.bench_function("srotg_coral_safe", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                let mut a = black_box(a0);
                let mut b = black_box(b0);
                let mut cval = 0.0f32;
                let mut sval = 0.0f32;

                srotg_safe(&mut a, &mut b, &mut cval, &mut sval);

                black_box((a, b, cval, sval));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("srotg_coral_neon", |b| {
        b.iter_batched_ref(
            || (),
            |_| {
                let mut a = black_box(a0);
                let mut b = black_box(b0);
                let mut cval = 0.0f32;
                let mut sval = 0.0f32;

                srotg_neon(&mut a, &mut b, &mut cval, &mut sval);

                black_box((a, b, cval, sval));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("srotg_blas", |b| {
        b.iter_batched_ref(
            || (),
            |_| unsafe {
                let mut a = black_box(a0);
                let mut b = black_box(b0);
                let mut cval = 0.0f32;
                let mut sval = 0.0f32;

                black_box(cblas_srotg(
                    black_box(&mut a as *mut f32),
                    black_box(&mut b as *mut f32),
                    black_box(&mut cval as *mut f32),
                    black_box(&mut sval as *mut f32),
                ));

                black_box((a, b, cval, sval));
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

criterion_group!(
    benches,
    bench_srot_big,
    bench_srotm_big,
    bench_srotmg_big,
    bench_srotg_big,
    bench_srot_sweep,
);
criterion_main!(benches);

