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
use coral::level1::scopy as scopy_safe;
use coral_aarch64::level1::scopy as scopy_neon;

pub fn scopy_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("scopy_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let incy = 1;

        let x_init = make_strided_vec(n, incx);
        let y_init = make_strided_vec(n, incy);

        group.throughput(Throughput::Bytes(bytes(n, 2)));

        group.bench_with_input(
            BenchmarkId::new("scopy_coral", n),
            &n,
            |b, &_n| {
                let xsafe = x_init.clone();
                let mut ysafe = y_init.clone();

                b.iter(|| {
                    let xcoral_aarch64 = make_view_ref(&xsafe, n, incx);
                    let ycoral_aarch64 = make_view_mut(&mut ysafe, n, incy);
                    black_box(scopy_safe(black_box(xcoral_aarch64), black_box(ycoral_aarch64)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scopy_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let xneon = x_init.clone();
                let mut yneon = y_init.clone();

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
            |b, &_n| {
                let xblas = x_init.clone();
                let mut yblas = y_init.clone();

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

