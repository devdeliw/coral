mod common; 
use common::{ 
    make_view_mut, 
    make_matview_ref, 
    bytes_decimal, 
    make_strided_vec, 
    make_triangular_mat, 
}; 

use criterion::{
    criterion_main, 
    criterion_group, 
    Criterion,
    Throughput,
};

use blas_src as _; 
use cblas_sys::{cblas_strsv, CBLAS_TRANSPOSE, CBLAS_LAYOUT, CBLAS_DIAG, CBLAS_UPLO}; 
use coral::level2::strsv as strsv_neon; 
use coral::enums::{
    CoralDiagonal as NeonDiagonal,
    CoralTranspose as NeonTranspose, 
    CoralTriangular as NeonTriangular, 
};

use coral_safe::level2::strsv as strsv_safe; 
use coral_safe::types::{CoralTriangular, CoralTranspose, CoralDiagonal};

pub fn strlsv_n(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(
        CoralTriangular::Lower, 
        CoralDiagonal::NonUnit, 
        n, lda 
    ); 

    let mut xsafe = xbuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut xblas = xbuf.clone(); 

    let aview = make_matview_ref(&abuf, n, n, lda);

    let mut group = c.benchmark_group("strlsv_n"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5))); 

    group.bench_function("strlsv_n_coral_safe", |b| { 
        b.iter(|| { 
            let xview = make_view_mut(&mut xsafe, n, incx); 
            strsv_safe (
                CoralTriangular::Lower, 
                CoralTranspose::NoTrans,
                CoralDiagonal::NonUnit, 
                aview,
                xview
            ); 
        }); 
    }); 

    group.bench_function("strlsv_n_coral_neon", |b| { 
        b.iter(|| { 
            strsv_neon (
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
    }); 

    group.bench_function("strlsv_n_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_strsv (
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
    }); 
}

pub fn strlsv_t(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(
        CoralTriangular::Lower, 
        CoralDiagonal::NonUnit, 
        n, lda 
    ); 

    let mut xsafe = xbuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut xblas = xbuf.clone(); 

    let aview = make_matview_ref(&abuf, n, n, lda);

    let mut group = c.benchmark_group("strlsv_t"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5))); 

    group.bench_function("strlsv_t_coral_safe", |b| { 
        b.iter(|| { 
            let xview = make_view_mut(&mut xsafe, n, incx); 
            strsv_safe (
                CoralTriangular::Lower, 
                CoralTranspose::Trans,
                CoralDiagonal::NonUnit, 
                aview,
                xview
            ); 
        }); 
    }); 

    group.bench_function("strlsv_t_coral_neon", |b| { 
        b.iter(|| { 
            strsv_neon (
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
    }); 

    group.bench_function("strlsv_t_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_strsv (
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
    }); 
}

pub fn strusv_n(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(
        CoralTriangular::Upper, 
        CoralDiagonal::NonUnit, 
        n, lda 
    ); 

    let mut xsafe = xbuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut xblas = xbuf.clone(); 

    let aview = make_matview_ref(&abuf, n, n, lda);

    let mut group = c.benchmark_group("strusv_n"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5))); 

    group.bench_function("strusv_n_coral_safe", |b| { 
        b.iter(|| { 
            let xview = make_view_mut(&mut xsafe, n, incx); 
            strsv_safe (
                CoralTriangular::Upper, 
                CoralTranspose::NoTrans,
                CoralDiagonal::NonUnit, 
                aview,
                xview
            ); 
        }); 
    }); 

    group.bench_function("strusv_n_coral_neon", |b| { 
        b.iter(|| { 
            strsv_neon (
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
    }); 

    group.bench_function("strusv_n_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_strsv (
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
    }); 
}

pub fn strusv_t(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_triangular_mat(
        CoralTriangular::Upper, 
        CoralDiagonal::NonUnit, 
        n, lda 
    ); 

    let mut xsafe = xbuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut xblas = xbuf.clone(); 

    let aview = make_matview_ref(&abuf, n, n, lda);

    let mut group = c.benchmark_group("strusv_t"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5))); 

    group.bench_function("strusv_t_coral_safe", |b| { 
        b.iter(|| { 
            let xview = make_view_mut(&mut xsafe, n, incx); 
            strsv_safe (
                CoralTriangular::Upper, 
                CoralTranspose::Trans,
                CoralDiagonal::NonUnit, 
                aview,
                xview
            ); 
        }); 
    }); 

    group.bench_function("strusv_t_coral_neon", |b| { 
        b.iter(|| { 
            strsv_neon (
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
    }); 

    group.bench_function("strusv_t_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_strsv (
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
    }); 
}

criterion_group!(benches, strlsv_n, strlsv_t, strusv_n, strusv_t); 
criterion_main!(benches);
