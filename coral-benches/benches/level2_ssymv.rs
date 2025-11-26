mod common; 
use common::{
    make_view_ref, 
    make_view_mut, 
    make_matview_ref, 
    bytes_decimal,
    make_strided_vec, 
    make_strided_mat, 
}; 

use criterion::{
    criterion_main, 
    criterion_group, 
    Criterion, 
    Throughput, 
}; 

use blas_src as _; 
use cblas_sys::{cblas_ssymv, CBLAS_LAYOUT, CBLAS_UPLO};
use coral::level2::ssymv as ssymv_safe; 
use coral::types::CoralTriangular;

pub fn ssymv_upper_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let lda = n;

    let incx = 1; 
    let incy = 1; 

    let alpha = 3.1415; 
    let beta  = 2.1828; 

    let abuf = make_strided_mat(n, n, lda);
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let acoral_aarch64 = make_matview_ref(&abuf, n, n, lda);
    let xcoral_aarch64 = make_view_ref(&xbuf, n, incx); 

    let mut group = c.benchmark_group("ssymv_upper_contiguous"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n) as f32, 0.5)));

    group.bench_function("ssymv_upper_coral", |b| { 
        b.iter(|| { 
            let ycoral_aarch64 = make_view_mut(&mut ybuf, n, incy);   
            ssymv_safe(CoralTriangular::Upper, alpha, beta, acoral_aarch64, xcoral_aarch64, ycoral_aarch64); 
        }); 
    }); 

    group.bench_function("ssymv_upper_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_ssymv (
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
    }); 
}

pub fn ssymv_lower_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let lda = n;

    let incx = 1; 
    let incy = 1; 

    let alpha = 3.1415; 
    let beta  = 2.1828; 

    let abuf = make_strided_mat(n, n, lda);
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let acoral_aarch64 = make_matview_ref(&abuf, n, n, lda);
    let xcoral_aarch64 = make_view_ref(&xbuf, n, incx); 

    let mut group = c.benchmark_group("ssymv_lower_contiguous"); 
    group.throughput(Throughput::Bytes(bytes_decimal((n * n) as f32, 0.5))); 

    group.bench_function("ssymv_lower_coral", |b| { 
        b.iter(|| { 
            let ycoral_aarch64 = make_view_mut(&mut ybuf, n, incy);   
            ssymv_safe(CoralTriangular::Lower, alpha, beta, acoral_aarch64, xcoral_aarch64, ycoral_aarch64); 
        }); 
    }); 

    group.bench_function("ssymv_lower_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_ssymv (
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
    }); 
}

criterion_group!(benches, ssymv_upper_contiguous, ssymv_lower_contiguous); 
criterion_main!(benches); 
