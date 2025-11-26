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
use cblas_sys::{cblas_ssyr, CBLAS_LAYOUT, CBLAS_UPLO}; 
use coral_aarch64::level2::ssyr as ssyr_neon; 
use coral_aarch64::enums::CoralTriangular as NeonTriangular; 
use coral::level2::ssyr as ssyr_safe; 
use coral::types::CoralTriangular; 


pub fn ssyr_upper_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 
    let alpha = 3.1415926; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_strided_mat(n, n, lda);

    let mut asafe = abuf.clone(); 
    let mut aneon = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = make_view_ref(&xbuf, n, incx); 

    let mut group = c.benchmark_group("ssyr_upper_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n * n, 1))); 

    group.bench_function("ssyr_upper_coral", |b| { 
        b.iter(|| { 
            let aview = make_matview_mut(&mut asafe, n, n, lda); 
            ssyr_safe(CoralTriangular::Upper, alpha, aview, xview); 
        }); 
    }); 

    group.bench_function("ssyr_upper_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            ssyr_neon (
                NeonTriangular::UpperTriangular, 
                n, 
                alpha,
                &xbuf, 
                incx, 
                &mut aneon, 
                lda, 
            ); 
        }); 
    }); 

    group.bench_function("ssyr_upper_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_ssyr (
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
    });     
}

pub fn ssyr_lower_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1; 
    let lda = n; 
    let alpha = 3.1415926; 

    let xbuf = make_strided_vec(n, incx); 
    let abuf = make_strided_mat(n, n, lda);

    let mut asafe = abuf.clone(); 
    let mut aneon = abuf.clone(); 
    let mut ablas = abuf.clone(); 

    let xview = make_view_ref(&xbuf, n, incx); 

    let mut group = c.benchmark_group("ssyr_lower_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n * n, 1))); 

    group.bench_function("ssyr_lower_coral", |b| { 
        b.iter(|| { 
            let aview = make_matview_mut(&mut asafe, n, n, lda); 
            ssyr_safe(CoralTriangular::Lower, alpha, aview, xview); 
        }); 
    }); 

    group.bench_function("ssyr_lower_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            ssyr_neon (
                NeonTriangular::LowerTriangular, 
                n, 
                alpha,
                &xbuf, 
                incx, 
                &mut aneon, 
                lda, 
            ); 
        }); 
    }); 

    group.bench_function("ssyr_lower_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_ssyr (
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
    });     
}

criterion_group!(benches, ssyr_upper_contiguous, ssyr_lower_contiguous); 
criterion_main!(benches); 
