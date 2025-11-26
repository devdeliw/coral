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
use cblas_sys::{cblas_ssyr, CBLAS_LAYOUT, CBLAS_UPLO};
use coral_aarch64::level2::ssyr as ssyr_neon;
use coral_aarch64::enums::CoralTriangular as NeonTriangular;
use coral::level2::ssyr as ssyr_safe;
use coral::types::CoralTriangular;

pub fn ssyr_upper_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssyr_upper_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let lda = n;
        let alpha: f32 = 3.1415926;

        group.throughput(Throughput::Bytes(bytes(n * n, 1)));

        group.bench_with_input(
            BenchmarkId::new("ssyr_upper_coral", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let abuf = make_strided_mat(n, n, lda);

                let xview = make_view_ref(&xbuf, n, incx);
                let mut asafe = abuf.clone();

                b.iter(|| {
                    let aview = make_matview_mut(&mut asafe, n, n, lda);
                    ssyr_safe(CoralTriangular::Upper, alpha, aview, xview);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssyr_upper_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut aneon = make_strided_mat(n, n, lda);

                b.iter(|| {
                    ssyr_neon(
                        NeonTriangular::UpperTriangular,
                        n,
                        alpha,
                        &xbuf,
                        incx,
                        &mut aneon,
                        lda,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssyr_upper_cblas", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut ablas = make_strided_mat(n, n, lda);

                b.iter(|| unsafe {
                    cblas_ssyr(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasUpper,
                        n as i32,
                        alpha,
                        xbuf.as_ptr(),
                        incx as i32,
                        ablas.as_mut_ptr(),
                        lda as i32,
                    );
                });
            },
        );
    }

    group.finish();
}

pub fn ssyr_lower_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssyr_lower_contiguous_sweep");

    for &n in SIZES {
        let incx = 1;
        let lda = n;
        let alpha: f32 = 3.1415926;

        group.throughput(Throughput::Bytes(bytes(n * n, 1)));

        group.bench_with_input(
            BenchmarkId::new("ssyr_lower_coral", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let abuf = make_strided_mat(n, n, lda);

                let xview = make_view_ref(&xbuf, n, incx);
                let mut asafe = abuf.clone();

                b.iter(|| {
                    let aview = make_matview_mut(&mut asafe, n, n, lda);
                    ssyr_safe(CoralTriangular::Lower, alpha, aview, xview);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssyr_lower_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut aneon = make_strided_mat(n, n, lda);

                b.iter(|| {
                    ssyr_neon(
                        NeonTriangular::LowerTriangular,
                        n,
                        alpha,
                        &xbuf,
                        incx,
                        &mut aneon,
                        lda,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ssyr_lower_cblas", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut ablas = make_strided_mat(n, n, lda);

                b.iter(|| unsafe {
                    cblas_ssyr(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasLower,
                        n as i32,
                        alpha,
                        xbuf.as_ptr(),
                        incx as i32,
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
    ssyr_upper_contiguous_sweep,
    ssyr_lower_contiguous_sweep
);
criterion_main!(benches);

