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
use coral_safe::level1::sscal as sscal_safe;
use coral::level1::sscal as sscal_neon;

pub fn sscal_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sscal_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let alpha = 3.1415926535_f32;

        let x = make_strided_vec(n, incx);
        let xsafe_init = x.clone();
        let xneon_init = x.clone();
        let xblas_init = x;

        group.throughput(Throughput::Bytes(bytes(n, 2)));

        group.bench_with_input(
            BenchmarkId::new("sscal_coral_safe", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let alpha = alpha;
                let mut xsafe = xsafe_init;

                b.iter(|| {
                    let xcoral = make_view_mut(&mut xsafe, n, incx);
                    black_box(sscal_safe(black_box(alpha), black_box(xcoral)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sscal_coral_neon", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let alpha = alpha;
                let mut xneon = xneon_init;

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
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let alpha = alpha;
                let mut xblas = xblas_init;

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

