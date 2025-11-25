mod common;
use common::{make_strided_vec, bytes, make_view_ref, make_view_mut, SIZES};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
    black_box,
};

use blas_src as _;
use cblas_sys::cblas_scopy;
use coral_safe::level1::scopy as scopy_safe;
use coral::level1::scopy as scopy_neon;

pub fn scopy_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("scopy_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let incy = 1;

        let x = make_strided_vec(n, incx);
        let y = make_strided_vec(n, incy);

        let xsafe_init = x.clone();
        let xneon_init = x.clone();
        let xblas_init = x;

        let ysafe_init = y.clone();
        let yneon_init = y.clone();
        let yblas_init = y;

        group.throughput(Throughput::Bytes(bytes(n, 2)));

        group.bench_with_input(
            BenchmarkId::new("scopy_coral_safe", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let incy = incy;

                let xsafe = xsafe_init;
                let mut ysafe = ysafe_init;

                b.iter(|| {
                    let xcoral = make_view_ref(&xsafe, n, incx);
                    let ycoral = make_view_mut(&mut ysafe, n, incy);
                    black_box(scopy_safe(black_box(xcoral), black_box(ycoral)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scopy_coral_neon", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let incy = incy;

                let xneon = xneon_init;
                let mut yneon = yneon_init;

                b.iter(|| {
                    black_box(scopy_neon(
                        black_box(n),
                        black_box(&xneon),
                        black_box(incx),
                        black_box(&mut yneon),
                        black_box(incy),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scopy_cblas", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let incy = incy;

                let xblas = xblas_init;
                let mut yblas = yblas_init;

                b.iter(|| unsafe {
                    black_box(cblas_scopy(
                        black_box(n as i32),
                        black_box(xblas.as_ptr()),
                        black_box(incx as i32),
                        black_box(yblas.as_mut_ptr()),
                        black_box(incy as i32),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, scopy_contiguous_sweep);
criterion_main!(benches);

