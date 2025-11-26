mod common;
use common::{
    make_matview_mut,
    make_view_ref,
    bytes,
    make_strided_vec,
    make_strided_mat,
    SIZES,
};

use criterion::{
    criterion_main,
    criterion_group,
    BenchmarkId,
    Criterion,
    Throughput,
};

use blas_src as _;
use cblas_sys::{cblas_ssyr2, CBLAS_LAYOUT, CBLAS_UPLO};
use coral_aarch64::level2::ssyr2 as ssyr2_neon;
use coral_aarch64::enums::CoralTriangular as NeonTriangular;
use coral::level2::ssyr2 as ssyr2_safe;
use coral::types::CoralTriangular;

pub fn ssyr2_upper_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssyr2_upper_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let incy = 1;
        let lda = n;
        let alpha: f32 = 3.1415926;

        group.throughput(Throughput::Bytes(bytes(n * n, 1)));

        group.bench_with_input(
            BenchmarkId::new("ssyr2_upper_coral", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let ybuf = make_strided_vec(n, incy);
                let abuf = make_strided_mat(n, n, lda);

                let xview = make_view_ref(&xbuf, n, incx);
                let yview = make_view_ref(&ybuf, n, incy);
                let mut asafe = abuf.clone();

                b.iter(|| {
                    let aview = make_matview_mut(&mut asafe, n, n, lda);
                    ssyr2_safe(CoralTriangular::Upper, alpha, aview, xview, yview);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssyr2_upper_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let ybuf = make_strided_vec(n, incy);
                let mut aneon = make_strided_mat(n, n, lda);

                b.iter(|| {
                    ssyr2_neon(
                        NeonTriangular::UpperTriangular,
                        n,
                        alpha,
                        &xbuf,
                        incx,
                        &ybuf,
                        incy,
                        &mut aneon,
                        lda,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssyr2_upper_cblas", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let ybuf = make_strided_vec(n, incy);
                let mut ablas = make_strided_mat(n, n, lda);

                b.iter(|| unsafe {
                    cblas_ssyr2(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasUpper,
                        n as i32,
                        alpha,
                        xbuf.as_ptr(),
                        incx as i32,
                        ybuf.as_ptr(),
                        incy as i32,
                        ablas.as_mut_ptr(),
                        lda as i32,
                    );
                });
            },
        );
    }

    group.finish();
}

pub fn ssyr2_lower_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssyr2_lower_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let incy = 1;
        let lda = n;
        let alpha: f32 = 3.1415926;

        group.throughput(Throughput::Bytes(bytes(n * n, 1)));

        group.bench_with_input(
            BenchmarkId::new("ssyr2_lower_coral", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let ybuf = make_strided_vec(n, incy);
                let abuf = make_strided_mat(n, n, lda);

                let xview = make_view_ref(&xbuf, n, incx);
                let yview = make_view_ref(&ybuf, n, incy);
                let mut asafe = abuf.clone();

                b.iter(|| {
                    let aview = make_matview_mut(&mut asafe, n, n, lda);
                    ssyr2_safe(CoralTriangular::Lower, alpha, aview, xview, yview);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssyr2_lower_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let ybuf = make_strided_vec(n, incy);
                let mut aneon = make_strided_mat(n, n, lda);

                b.iter(|| {
                    ssyr2_neon(
                        NeonTriangular::LowerTriangular,
                        n,
                        alpha,
                        &xbuf,
                        incx,
                        &ybuf,
                        incy,
                        &mut aneon,
                        lda,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssyr2_lower_cblas", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let ybuf = make_strided_vec(n, incy);
                let mut ablas = make_strided_mat(n, n, lda);

                b.iter(|| unsafe {
                    cblas_ssyr2(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasLower,
                        n as i32,
                        alpha,
                        xbuf.as_ptr(),
                        incx as i32,
                        ybuf.as_ptr(),
                        incy as i32,
                        ablas.as_mut_ptr(),
                        lda as i32,
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    ssyr2_upper_contiguous_sweep,
    ssyr2_lower_contiguous_sweep
);
criterion_main!(benches);

