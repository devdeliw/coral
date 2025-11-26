mod common; 
use common::{make_strided_vec, bytes, make_view_mut}; 

use criterion::{
    criterion_main, 
    criterion_group, 
    Criterion, 
    Throughput, 
    black_box, 
};

use blas_src as _; 
use cblas_sys::cblas_sscal; 
use coral::level1::sscal as sscal_safe;
use coral_aarch64::level1::sscal as sscal_neon;

pub fn sscal_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 1; 
    let alpha = 3.1415926535; 

    let xbuf = make_strided_vec(n, incx); 
    
    let mut xsafe = xbuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut xblas = xbuf.clone(); 

    let mut group = c.benchmark_group("sscal_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n, 2))); 

    group.bench_function("sscal_coral", |b| { 
        b.iter(|| { 
            let xcoral_aarch64 = make_view_mut(&mut xsafe, n, incx); 
            black_box(sscal_safe(black_box(alpha), black_box(xcoral_aarch64))); 
        }); 
    }); 

    group.bench_function("sscal_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box( sscal_neon ( 
                black_box(n), 
                black_box(alpha), 
                black_box(&mut xneon), 
                black_box(incx) 
            )); 
        }); 
    }); 

    group.bench_function("sscal_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box( cblas_sscal ( 
                black_box(n as i32), 
                black_box(alpha), 
                black_box(xblas.as_mut_ptr()), 
                black_box(incx as i32)
            )); 
        }); 
    }); 
}

pub fn sscal_strided(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 2; 
    let alpha = 3.1415926535; 

    let xbuf = make_strided_vec(n, incx); 
    
    let mut xsafe = xbuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut xblas = xbuf.clone(); 

    let mut group = c.benchmark_group("sscal_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 2))); 

    group.bench_function("sscal_coral", |b| { 
        b.iter(|| { 
            let xcoral_aarch64 = make_view_mut(&mut xsafe, n, incx); 
            black_box(sscal_safe(black_box(alpha), black_box(xcoral_aarch64))); 
        }); 
    }); 

    group.bench_function("sscal_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box( sscal_neon ( 
                black_box(n), 
                black_box(alpha), 
                black_box(&mut xneon), 
                black_box(incx) 
            )); 
        }); 
    }); 

    group.bench_function("sscal_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box( cblas_sscal ( 
                black_box(n as i32), 
                black_box(alpha), 
                black_box(xblas.as_mut_ptr()), 
                black_box(incx as i32)
            )); 
        }); 
    }); 
}

criterion_group!(benches, sscal_contiguous, sscal_strided); 
criterion_main!(benches); 
