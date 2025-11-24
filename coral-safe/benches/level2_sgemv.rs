mod common; 
use common::{
    make_view_ref, 
    make_view_mut, 
    make_matview_ref, 
    bytes,
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
use cblas_sys::{cblas_sgemv, CBLAS_TRANSPOSE, CBLAS_LAYOUT};
use coral::level2::sgemv as sgemv_neon; 
use coral::enums::CoralTranspose as CoralNeonTranspose; 
use coral_safe::level2::sgemv as sgemv_safe; 
use coral_safe::types::CoralTranspose;

pub fn sgemv_n_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let m = 1024; 
    let lda = m;

    let incx = 1; 
    let incy = 1; 

    let alpha = 3.1415; 
    let beta  = 2.71828; 

    let abuf = make_strided_mat(m, n, lda);
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(m, incy); 

    let acoral = make_matview_ref(&abuf, m, n, lda);
    let xcoral = make_view_ref(&xbuf, n, incx); 

    let mut group = c.benchmark_group("sgemv_n_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n * m, 1))); 

    group.bench_function("sgemv_n_coral_safe", |b| { 
        b.iter(|| { 
            let ycoral = make_view_mut(&mut ybuf, m, incy);   
            sgemv_safe(CoralTranspose::NoTrans, alpha, beta, acoral, xcoral, ycoral); 
        }); 
    }); 

    group.bench_function("sgemv_n_coral_neon", |b| { 
        b.iter(|| { 
            sgemv_neon (
                CoralNeonTranspose::NoTranspose,
                m, 
                n, 
                alpha, 
                &abuf, 
                lda, 
                &xbuf, 
                incx, 
                beta, 
                &mut ybuf, 
                incy, 
            ); 
        }); 
    });

    group.bench_function("sgemv_n_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_sgemv (
                CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans, 
                m as i32, 
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

pub fn sgemv_t_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let m = 1024; 
    let lda = m;

    let incx = 1; 
    let incy = 1; 

    let alpha = 3.1415; 
    let beta  = 2.1828; 

    let abuf = make_strided_mat(m, n, lda);
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(m, incy); 

    let acoral = make_matview_ref(&abuf, m, n, lda);
    let xcoral = make_view_ref(&xbuf, n, incx); 

    let mut group = c.benchmark_group("sgemv_t_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n * m, 1))); 

    group.bench_function("sgemv_t_coral_safe", |b| { 
        b.iter(|| { 
            let ycoral = make_view_mut(&mut ybuf, m, incy);   
            sgemv_safe(CoralTranspose::Trans, alpha, beta, acoral, xcoral, ycoral); 
        }); 
    }); 

    group.bench_function("sgemv_t_coral_neon", |b| { 
        b.iter(|| { 
            sgemv_neon (
                CoralNeonTranspose::Transpose,
                m, 
                n, 
                alpha, 
                &abuf, 
                lda, 
                &xbuf, 
                incx, 
                beta, 
                &mut ybuf, 
                incy, 
            ); 
        }); 
    });

    group.bench_function("sgemv_t_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_sgemv (
                CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasTrans, 
                m as i32, 
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

criterion_group!(benches, sgemv_n_contiguous, sgemv_t_contiguous); 
criterion_main!(benches); 
