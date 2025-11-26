mod common;
use common::{make_strided_vec, bytes, make_view_mut, SIZES};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
    black_box,
};

use blas_src as _;
use cblas_sys::cblas_sscal;
use coral::level1::sscal as sscal_safe;
use coral_aarch64::level1::sscal as sscal_neon;

pub fn sscal_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sscal_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let alpha = 3.1415926535_f32;

        let x_init = make_strided_vec(n, incx);

        group.throughput(Throughput::Bytes(bytes(n, 2)));

        group.bench_with_input(
            BenchmarkId::new("sscal_coral", n),
            &n,
            |b, &_n| {
                let mut xsafe = x_init.clone();

                b.iter(|| {
                    let xcoral_aarch64 = make_view_mut(&mut xsafe, n, incx);
                    black_box(sscal_safe(black_box(alpha), black_box(xcoral_aarch64)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sscal_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let mut xneon = x_init.clone();

                b.iter(|| {
                    black_box(sscal_neon(
                        black_box(n),
                        black_box(alpha),
                        black_box(&mut xneon),
                        black_box(incx),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sscal_cblas", n),
            &n,
            |b, &_n| {
                let mut xblas = x_init.clone();

                b.iter(|| unsafe {
                    black_box(cblas_sscal(
                        black_box(n as i32),
                        black_box(alpha),
                        black_box(xblas.as_mut_ptr()),
                        black_box(incx as i32),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, sscal_contiguous_sweep);
criterion_main!(benches);

