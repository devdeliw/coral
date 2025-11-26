mod common;
use common::{
    make_view_ref,
    make_view_mut,
    make_matview_ref,
    bytes_decimal,
    make_strided_vec,
    make_strided_mat,
    SIZES,
};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};

use blas_src as _;
use cblas_sys::{cblas_ssymv, CBLAS_LAYOUT, CBLAS_UPLO};
use coral::level2::ssymv as ssymv_safe;
use coral::types::CoralTriangular;

pub fn ssymv_upper_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssymv_upper_contiguous_sweep");

    for &n in SIZES {
        let lda = n;
        let incx = 1;
        let incy = 1;
        let alpha: f32 = 3.1415;
        let beta: f32 = 2.1828;

        group.throughput(Throughput::Bytes(bytes_decimal((n * n) as f32, 0.5)));

        group.bench_with_input(
            BenchmarkId::new("ssymv_upper_coral", n),
            &n,
            |b, &_n| {
                let abuf = make_strided_mat(n, n, lda);
                let xbuf = make_strided_vec(n, incx);
                let mut ybuf = make_strided_vec(n, incy);

                let acoral_aarch64 = make_matview_ref(&abuf, n, n, lda);
                let xcoral_aarch64 = make_view_ref(&xbuf, n, incx);

                b.iter(|| {
                    let ycoral_aarch64 = make_view_mut(&mut ybuf, n, incy);
                    ssymv_safe(CoralTriangular::Upper, alpha, beta, acoral_aarch64, xcoral_aarch64, ycoral_aarch64);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssymv_upper_cblas", n),
            &n,
            |b, &_n| {
                let abuf = make_strided_mat(n, n, lda);
                let xbuf = make_strided_vec(n, incx);
                let mut ybuf = make_strided_vec(n, incy);

                b.iter(|| unsafe {
                    cblas_ssymv(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasUpper,
                        n as i32,
                        alpha,
                        abuf.as_ptr(),
                        lda as i32,
                        xbuf.as_ptr(),
                        incx as i32,
                        beta,
                        ybuf.as_mut_ptr(),
                        incy as i32,
                    );
                });
            },
        );
    }

    group.finish();
}

pub fn ssymv_lower_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssymv_lower_contiguous_sweep");

    for &n in SIZES {
        let lda = n;
        let incx = 1;
        let incy = 1;
        let alpha: f32 = 3.1415;
        let beta: f32 = 2.1828;

        group.throughput(Throughput::Bytes(bytes_decimal((n * n) as f32, 0.5)));

        group.bench_with_input(
            BenchmarkId::new("ssymv_lower_coral", n),
            &n,
            |b, &_n| {
                let abuf = make_strided_mat(n, n, lda);
                let xbuf = make_strided_vec(n, incx);
                let mut ybuf = make_strided_vec(n, incy);

                let acoral_aarch64 = make_matview_ref(&abuf, n, n, lda);
                let xcoral_aarch64 = make_view_ref(&xbuf, n, incx);

                b.iter(|| {
                    let ycoral_aarch64 = make_view_mut(&mut ybuf, n, incy);
                    ssymv_safe(CoralTriangular::Lower, alpha, beta, acoral_aarch64, xcoral_aarch64, ycoral_aarch64);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssymv_lower_cblas", n),
            &n,
            |b, &_n| {
                let abuf = make_strided_mat(n, n, lda);
                let xbuf = make_strided_vec(n, incx);
                let mut ybuf = make_strided_vec(n, incy);

                b.iter(|| unsafe {
                    cblas_ssymv(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasLower,
                        n as i32,
                        alpha,
                        abuf.as_ptr(),
                        lda as i32,
                        xbuf.as_ptr(),
                        incx as i32,
                        beta,
                        ybuf.as_mut_ptr(),
                        incy as i32,
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    ssymv_upper_contiguous_sweep,
    ssymv_lower_contiguous_sweep
);
criterion_main!(benches);

