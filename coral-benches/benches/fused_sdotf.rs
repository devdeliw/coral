mod common; 
use common::{
    make_strided_mat,
    make_strided_vec, 
    make_view_ref, 
    make_view_mut,
    make_matview_ref, 
    bytes
}; 

use criterion::{
    criterion_main, 
    criterion_group, 
    Criterion,
    Throughput, 
    black_box
};

use blas_src as _; 
use cblas_sys::{cblas_sgemv, CBLAS_TRANSPOSE, CBLAS_LAYOUT}; 
use coral::fused::sdotf;
use coral_aarch64::enums::CoralTranspose; 
use coral_aarch64::level2::sgemv;

pub fn saxpyf_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1; 
    let incy = 1; 
    let lda = n; 

    // saxpyf equivalent to sgemv with 
    // alpha, beta = 1.0
    let alpha = 1.0; 
    let beta  = 1.0; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let xcoral_aarch64 = make_view_ref(&xbuf, n, incx); 
    let acoral_aarch64 = make_matview_ref(&abuf, n, n, lda);
    
    let mut group = c.benchmark_group("sdotf_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n*n + 3*n, 1))); 

    group.bench_function("sdotf_coral", |b| { 
        b.iter(|| { 
            let ycoral_aarch64 = make_view_mut(&mut ybuf, n, incy); 
            
            sdotf(
                black_box(acoral_aarch64), 
                black_box(xcoral_aarch64),
                black_box(ycoral_aarch64)
            ); 
        }); 
    });

    group.bench_function("sdotf_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            sgemv ( 
                black_box(CoralTranspose::Transpose), 
                black_box(n), 
                black_box(n), 
                black_box(alpha), 
                black_box(&abuf), 
                black_box(lda), 
                black_box(&xbuf), 
                black_box(incx), 
                black_box(beta), 
                black_box(&mut ybuf), 
                black_box(incy)
            ); 
        });
    }); 

    group.bench_function("sdotf_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_sgemv ( 
                black_box(CBLAS_LAYOUT::CblasColMajor), 
                black_box(CBLAS_TRANSPOSE::CblasTrans), 
                black_box(n as i32), 
                black_box(n as i32), 
                black_box(alpha), 
                black_box(abuf.as_ptr()), 
                black_box(lda as i32), 
                black_box(xbuf.as_ptr()), 
                black_box(incx as i32), 
                black_box(beta), 
                black_box(ybuf.as_mut_ptr()), 
                black_box(incy as i32), 
            );
        }); 
    }); 
}

pub fn saxpyf_strided(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 2; 
    let incy = 3; 
    let lda = n; 

    let alpha = 1.0; 
    let beta  = 1.0; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let xcoral_aarch64 = make_view_ref(&xbuf, n, incx); 
    let acoral_aarch64 = make_matview_ref(&abuf, n, n, lda);
    
    let mut group = c.benchmark_group("sdotf_strided"); 
    group.throughput(Throughput::Bytes(bytes(n*n + 3*n, 1))); 

    group.bench_function("sdotf_coral", |b| { 
        b.iter(|| { 
            let ycoral_aarch64 = make_view_mut(&mut ybuf, n, incy); 
            
            sdotf(
                black_box(acoral_aarch64), 
                black_box(xcoral_aarch64),
                black_box(ycoral_aarch64)
            ); 
        }); 
    });

    group.bench_function("sdotf_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            sgemv ( 
                black_box(CoralTranspose::Transpose), 
                black_box(n), 
                black_box(n), 
                black_box(alpha), 
                black_box(&abuf), 
                black_box(lda), 
                black_box(&xbuf), 
                black_box(incx), 
                black_box(beta), 
                black_box(&mut ybuf), 
                black_box(incy)
            );
        }); 
    });

    group.bench_function("sdotf_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_sgemv ( 
                black_box(CBLAS_LAYOUT::CblasColMajor), 
                black_box(CBLAS_TRANSPOSE::CblasTrans), 
                black_box(n as i32), 
                black_box(n as i32), 
                black_box(alpha), 
                black_box(abuf.as_ptr()), 
                black_box(lda as i32), 
                black_box(xbuf.as_ptr()), 
                black_box(incx as i32), 
                black_box(beta), 
                black_box(ybuf.as_mut_ptr()), 
                black_box(incy as i32), 
            ); 
        }); 
    }); 
}

criterion_group!(benches, saxpyf_contiguous, saxpyf_strided); 
criterion_main!(benches); 

