mod common;
use common::{make_strided_vec, bytes, make_view_ref, SIZES};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
    black_box,
};

use blas_src as _;
use cblas_sys::cblas_sdot;
use coral_safe::level1::sdot as sdot_safe;
use coral::level1::sdot as sdot_neon;

pub fn sdot_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdot_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let incy = 1;

        let xbuf: &'static [f32] = Box::leak(make_strided_vec(n, incx).into_boxed_slice());
        let ybuf: &'static [f32] = Box::leak(make_strided_vec(n, incy).into_boxed_slice());

        let xvec = make_view_ref(xbuf, n, incx);
        let yvec = make_view_ref(ybuf, n, incy);

        group.throughput(Throughput::Bytes(bytes(n, 2)));

        group.bench_with_input(
            BenchmarkId::new("sdot_coral_safe", n),
            &n,
            move |b, &_n| {
                b.iter(|| {
                    black_box(sdot_safe(black_box(xvec), black_box(yvec)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sdot_coral_neon", n),
            &n,
            move |b, &_n| {
                b.iter(|| {
                    black_box(sdot_neon(
                        black_box(n),
                        black_box(xbuf),
                        black_box(incx),
                        black_box(ybuf),
                        black_box(incy),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sdot_cblas", n),
            &n,
            move |b, &_n| {
                b.iter(|| unsafe {
                    black_box(cblas_sdot(
                        black_box(n as i32),
                        black_box(xbuf.as_ptr()),
                        black_box(incx as i32),
                        black_box(ybuf.as_ptr()),
                        black_box(incy as i32),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, sdot_contiguous_sweep);
criterion_main!(benches);

