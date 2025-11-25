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
use cblas_sys::cblas_srot;
use coral_safe::level1::srot as srot_safe;
use coral::level1::srot as srot_neon;

pub fn srot_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("srot_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let incy = 1;

        let c_val = 0.8_f32;
        let s_val = 0.6_f32;

        let x = make_strided_vec(n, incx);
        let y = make_strided_vec(n, incy);

        let xsafe_init = x.clone();
        let ysafe_init = y.clone();

        let xneon_init = x.clone();
        let yneon_init = y.clone();

        let xblas_init = x;
        let yblas_init = y;

        group.throughput(Throughput::Bytes(bytes(n, 4)));

        group.bench_with_input(
            BenchmarkId::new("srot_coral_safe", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let incy = incy;
                let c_val = c_val;
                let s_val = s_val;

                let mut xsafe = xsafe_init;
                let mut ysafe = ysafe_init;

                b.iter(|| {
                    let xcoral = make_view_mut(&mut xsafe, n, incx);
                    let ycoral = make_view_mut(&mut ysafe, n, incy);

                    black_box(srot_safe(
                        black_box(xcoral),
                        black_box(ycoral),
                        black_box(c_val),
                        black_box(s_val),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("srot_coral_neon", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let incy = incy;
                let c_val = c_val;
                let s_val = s_val;

                let mut xneon = xneon_init;
                let mut yneon = yneon_init;

                b.iter(|| {
                    black_box(srot_neon(
                        black_box(n),
                        black_box(&mut xneon),
                        black_box(incx),
                        black_box(&mut yneon),
                        black_box(incy),
                        black_box(c_val),
                        black_box(s_val),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("srot_cblas", n),
            &n,
            move |b, &_n| {
                let n = n;
                let incx = incx;
                let incy = incy;
                let c_val = c_val;
                let s_val = s_val;

                let mut xblas = xblas_init;
                let mut yblas = yblas_init;

                b.iter(|| unsafe {
                    black_box(cblas_srot(
                        black_box(n as i32),
                        black_box(xblas.as_mut_ptr()),
                        black_box(incx as i32),
                        black_box(yblas.as_mut_ptr()),
                        black_box(incy as i32),
                        black_box(c_val),
                        black_box(s_val),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, srot_contiguous_sweep);
criterion_main!(benches);

