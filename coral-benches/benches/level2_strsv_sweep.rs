mod common;
use common::{
    make_view_mut,
    make_matview_ref,
    bytes_decimal,
    make_strided_vec,
    make_triangular_mat,
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
use cblas_sys::{cblas_strsv, CBLAS_TRANSPOSE, CBLAS_LAYOUT, CBLAS_DIAG, CBLAS_UPLO};
use coral_aarch64::level2::strsv as strsv_neon;
use coral_aarch64::enums::{
    CoralDiagonal as NeonDiagonal,
    CoralTranspose as NeonTranspose,
    CoralTriangular as NeonTriangular,
};
use coral::level2::strsv as strsv_safe;
use coral::types::{CoralTriangular, CoralTranspose, CoralDiagonal};

pub fn strlsv_n_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("strlsv_n_sweep");

    for &n in SIZES {
        let incx = 1;
        let lda = n;

        group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5)));

        group.bench_with_input(
            BenchmarkId::new("strlsv_n_coral", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xsafe = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Lower,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );
                let aview = make_matview_ref(&abuf, n, n, lda);

                b.iter(|| {
                    let xview = make_view_mut(&mut xsafe, n, incx);
                    strsv_safe(
                        CoralTriangular::Lower,
                        CoralTranspose::NoTrans,
                        CoralDiagonal::NonUnit,
                        aview,
                        xview,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("strlsv_n_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xneon = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Lower,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );

                b.iter(|| {
                    strsv_neon(
                        NeonTriangular::LowerTriangular,
                        NeonTranspose::NoTranspose,
                        NeonDiagonal::NonUnitDiagonal,
                        n,
                        &abuf,
                        lda,
                        &mut xneon,
                        incx,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("strlsv_n_cblas", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xblas = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Lower,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );

                b.iter(|| unsafe {
                    cblas_strsv(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasLower,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        CBLAS_DIAG::CblasNonUnit,
                        n as i32,
                        abuf.as_ptr(),
                        lda as i32,
                        xblas.as_mut_ptr(),
                        incx as i32,
                    );
                });
            },
        );
    }

    group.finish();
}

pub fn strlsv_t_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("strlsv_t_sweep");

    for &n in SIZES {
        let incx = 1;
        let lda = n;

        group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5)));

        group.bench_with_input(
            BenchmarkId::new("strlsv_t_coral", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xsafe = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Lower,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );
                let aview = make_matview_ref(&abuf, n, n, lda);

                b.iter(|| {
                    let xview = make_view_mut(&mut xsafe, n, incx);
                    strsv_safe(
                        CoralTriangular::Lower,
                        CoralTranspose::Trans,
                        CoralDiagonal::NonUnit,
                        aview,
                        xview,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("strlsv_t_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xneon = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Lower,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );

                b.iter(|| {
                    strsv_neon(
                        NeonTriangular::LowerTriangular,
                        NeonTranspose::Transpose,
                        NeonDiagonal::NonUnitDiagonal,
                        n,
                        &abuf,
                        lda,
                        &mut xneon,
                        incx,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("strlsv_t_cblas", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xblas = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Lower,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );

                b.iter(|| unsafe {
                    cblas_strsv(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasLower,
                        CBLAS_TRANSPOSE::CblasTrans,
                        CBLAS_DIAG::CblasNonUnit,
                        n as i32,
                        abuf.as_ptr(),
                        lda as i32,
                        xblas.as_mut_ptr(),
                        incx as i32,
                    );
                });
            },
        );
    }

    group.finish();
}

pub fn strusv_n_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("strusv_n_sweep");

    for &n in SIZES {
        let incx = 1;
        let lda = n;

        group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5)));

        group.bench_with_input(
            BenchmarkId::new("strusv_n_coral", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xsafe = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Upper,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );
                let aview = make_matview_ref(&abuf, n, n, lda);

                b.iter(|| {
                    let xview = make_view_mut(&mut xsafe, n, incx);
                    strsv_safe(
                        CoralTriangular::Upper,
                        CoralTranspose::NoTrans,
                        CoralDiagonal::NonUnit,
                        aview,
                        xview,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("strusv_n_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xneon = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Upper,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );

                b.iter(|| {
                    strsv_neon(
                        NeonTriangular::UpperTriangular,
                        NeonTranspose::NoTranspose,
                        NeonDiagonal::NonUnitDiagonal,
                        n,
                        &abuf,
                        lda,
                        &mut xneon,
                        incx,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("strusv_n_cblas", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xblas = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Upper,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );

                b.iter(|| unsafe {
                    cblas_strsv(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasUpper,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        CBLAS_DIAG::CblasNonUnit,
                        n as i32,
                        abuf.as_ptr(),
                        lda as i32,
                        xblas.as_mut_ptr(),
                        incx as i32,
                    );
                });
            },
        );
    }

    group.finish();
}

pub fn strusv_t_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("strusv_t_sweep");

    for &n in SIZES {
        let incx = 1;
        let lda = n;

        group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5)));

        group.bench_with_input(
            BenchmarkId::new("strusv_t_coral", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xsafe = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Upper,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );
                let aview = make_matview_ref(&abuf, n, n, lda);

                b.iter(|| {
                    let xview = make_view_mut(&mut xsafe, n, incx);
                    strsv_safe(
                        CoralTriangular::Upper,
                        CoralTranspose::Trans,
                        CoralDiagonal::NonUnit,
                        aview,
                        xview,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("strusv_t_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xneon = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Upper,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );

                b.iter(|| {
                    strsv_neon(
                        NeonTriangular::UpperTriangular,
                        NeonTranspose::Transpose,
                        NeonDiagonal::NonUnitDiagonal,
                        n,
                        &abuf,
                        lda,
                        &mut xneon,
                        incx,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("strusv_t_cblas", n),
            &n,
            |b, &_n| {
                let xbuf = make_strided_vec(n, incx);
                let mut xblas = xbuf.clone();
                let abuf = make_triangular_mat(
                    CoralTriangular::Upper,
                    CoralDiagonal::NonUnit,
                    n,
                    lda,
                );

                b.iter(|| unsafe {
                    cblas_strsv(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_UPLO::CblasUpper,
                        CBLAS_TRANSPOSE::CblasTrans,
                        CBLAS_DIAG::CblasNonUnit,
                        n as i32,
                        abuf.as_ptr(),
                        lda as i32,
                        xblas.as_mut_ptr(),
                        incx as i32,
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    strlsv_n_sweep,
    strlsv_t_sweep,
    strusv_n_sweep,
    strusv_t_sweep
);
criterion_main!(benches);

