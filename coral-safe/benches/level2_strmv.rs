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
use cblas_sys::{cblas_strmv, CBLAS_TRANSPOSE, CBLAS_LAYOUT, CBLAS_DIAG, CBLAS_UPLO}; 
use coral::level2::strmv as strmv_neon; 
use coral::enums::{
    CoralDiagonal as NeonDiagonal,
    CoralTranspose as NeonTranspose, 
    CoralTriangular as NeonTriangular, 
};
use coral_safe::level2::strmv as strmv_safe; 
use coral_safe::types::{CoralTriangular, CoralTranspose, CoralDiagonal};

pub fn strlmv_n(c: &mut Criterion) { 
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

    let mut group = c.benchmark_group("strlmv_n"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5))); 

    group.bench_function("strlmv_n_coral_safe", |b| { 
        b.iter(|| { 
            let xview = make_view_mut(&mut xsafe, n, incx); 
            strmv_safe (
                CoralTriangular::Lower, 
                CoralTranspose::NoTrans,
                CoralDiagonal::NonUnit, 
                aview,
                xview
            ); 
        }); 
    }); 

    group.bench_function("strlmv_n_coral_neon", |b| { 
        b.iter(|| { 
            strmv_neon (
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

    group.bench_function("strlmv_n_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_strmv (
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

pub fn strlmv_t(c: &mut Criterion) { 
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

    let mut group = c.benchmark_group("strlmv_t"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5))); 

    group.bench_function("strlmv_t_coral_safe", |b| { 
        b.iter(|| { 
            let xview = make_view_mut(&mut xsafe, n, incx); 
            strmv_safe (
                CoralTriangular::Lower, 
                CoralTranspose::Trans,
                CoralDiagonal::NonUnit, 
                aview,
                xview
            ); 
        }); 
    }); 

    group.bench_function("strlmv_t_coral_neon", |b| { 
        b.iter(|| { 
            strmv_neon (
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

    group.bench_function("strlmv_t_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_strmv (
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

pub fn strumv_n(c: &mut Criterion) { 
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

    let mut group = c.benchmark_group("strumv_n"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5))); 

    group.bench_function("strumv_n_coral_safe", |b| { 
        b.iter(|| { 
            let xview = make_view_mut(&mut xsafe, n, incx); 
            strmv_safe (
                CoralTriangular::Upper, 
                CoralTranspose::NoTrans,
                CoralDiagonal::NonUnit, 
                aview,
                xview
            ); 
        }); 
    }); 

    group.bench_function("strumv_n_coral_neon", |b| { 
        b.iter(|| { 
            strmv_neon (
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

    group.bench_function("strumv_n_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_strmv (
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

pub fn strumv_t(c: &mut Criterion) { 
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

    let mut group = c.benchmark_group("strumv_t"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n + n) as f32, 0.5))); 

    group.bench_function("strumv_t_coral_safe", |b| { 
        b.iter(|| { 
            let xview = make_view_mut(&mut xsafe, n, incx); 
            strmv_safe (
                CoralTriangular::Upper, 
                CoralTranspose::Trans,
                CoralDiagonal::NonUnit, 
                aview,
                xview
            ); 
        }); 
    }); 

    group.bench_function("strumv_t_coral_neon", |b| { 
        b.iter(|| { 
            strmv_neon (
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

    group.bench_function("strumv_t_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_strmv (
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

criterion_group!(benches, strlmv_n, strlmv_t, strumv_n, strumv_t); 
criterion_main!(benches);
