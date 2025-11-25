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
use cblas_sys::cblas_sswap;
use coral_safe::level1::sswap as sswap_safe;
use coral::level1::sswap as sswap_neon;

pub fn sswap_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sswap_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let incy = 1;

        let mut xsafe = make_strided_vec(n, incx);
        let mut ysafe = make_strided_vec(n, incy);

        let mut xneon = xsafe.clone();
        let mut yneon = ysafe.clone();

        let mut xblas = xsafe.clone();
        let mut yblas = ysafe.clone();

        group.throughput(Throughput::Bytes(bytes(n, 4)));

        group.bench_with_input(
            BenchmarkId::new("sswap_coral_safe", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let incy = incy;
                let mut xsafe = xsafe;
                let mut ysafe = ysafe;

                b.iter(|| {
                    let xcoral = make_view_mut(&mut xsafe, n, incx);
                    let ycoral = make_view_mut(&mut ysafe, n, incy);
                    black_box(sswap_safe(black_box(xcoral), black_box(ycoral)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sswap_coral_neon", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let incy = incy;
                let mut xneon = xneon;
                let mut yneon = yneon;

                b.iter(|| {
                    black_box(sswap_neon(
                        black_box(n),
                        black_box(&mut xneon),
                        black_box(incx),
                        black_box(&mut yneon),
                        black_box(incy),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sswap_cblas", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let incy = incy;
                let mut xblas = xblas;
                let mut yblas = yblas;

                b.iter(|| unsafe {
                    black_box(cblas_sswap(
                        black_box(n as i32),
                        black_box(xblas.as_mut_ptr()),
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

criterion_group!(benches, sswap_contiguous_sweep);
criterion_main!(benches);
