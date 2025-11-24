mod common; 
use common::{ 
    make_matview_mut, 
    make_view_ref, 
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
use cblas_sys::{cblas_ssyr2, CBLAS_LAYOUT, CBLAS_UPLO}; 
use coral::level2::ssyr2 as ssyr2_neon; 
use coral::enums::CoralTriangular as NeonTriangular; 
use coral_safe::level2::ssyr2 as ssyr2_safe; 
use coral_safe::types::CoralTriangular; 


pub fn ssyr2_upper_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1;
    let incy = 1; 
    let lda = n; 
    let alpha = 3.1415926; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy);
    let abuf = make_strided_mat(n, n, lda);

    let mut asafe = abuf.clone(); 
    let mut aneon = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = make_view_ref(&xbuf, n, incx); 
    let yview = make_view_ref(&ybuf, n, incy); 

    let mut group = c.benchmark_group("ssyr2_upper_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n * n, 1))); 

    group.bench_function("ssyr2_upper_coral_safe", |b| { 
        b.iter(|| { 
            let aview = make_matview_mut(&mut asafe, n, n, lda); 
            ssyr2_safe(CoralTriangular::Upper, alpha, aview, xview, yview); 
        }); 
    }); 

    group.bench_function("ssyr2_upper_coral_neon", |b| { 
        b.iter(|| { 
            ssyr2_neon (
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
    }); 

    group.bench_function("ssyr2_upper_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_ssyr2 (
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
    });     
}

pub fn ssyr2_lower_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1;
    let incy = 1; 
    let lda = n; 
    let alpha = 3.1415926; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 
    let abuf = make_strided_mat(n, n, lda);

    let mut asafe = abuf.clone(); 
    let mut aneon = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = make_view_ref(&xbuf, n, incx); 
    let yview = make_view_ref(&ybuf, n, incy); 

    let mut group = c.benchmark_group("ssyr2_lower_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n * n, 1)));

    group.bench_function("ssyr2_lower_coral_safe", |b| { 
        b.iter(|| { 
            let aview = make_matview_mut(&mut asafe, n, n, lda); 
            ssyr2_safe(CoralTriangular::Lower, alpha, aview, xview, yview); 
        }); 
    }); 

    group.bench_function("ssyr2_lower_coral_neon", |b| { 
        b.iter(|| { 
            ssyr2_neon (
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
    }); 

    group.bench_function("ssyr2_lower_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_ssyr2 (
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
    });     
}

criterion_group!(benches, ssyr2_upper_contiguous, ssyr2_lower_contiguous); 
criterion_main!(benches); 
