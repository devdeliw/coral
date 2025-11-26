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
use cblas_sys::cblas_srotm; 
use coral::level1::srotm as srotm_safe; 
use coral_aarch64::level1::srotm as srotm_neon; 

pub fn srotm_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 1;
    let incy = 1; 

    // flag = -1; full givens
    let param = [-1.0, 3.14, 3.14, 3.14, 3.14]; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 

    // look away before this blinds you
    let mut xsafe = xbuf.clone(); 
    let mut ysafe = ybuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut yneon = ybuf.clone(); 
    let mut xblas = xbuf.clone(); 
    let mut yblas = ybuf.clone(); 

    let mut group = c.benchmark_group("srotm_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n, 4))); 

    group.bench_function("srotm_coral", |b| { 
        b.iter(|| { 
            // includes creation time; neglible 
            let xcoral_aarch64 = make_view_mut(&mut xsafe, n, incx); 
            let ycoral_aarch64 = make_view_mut(&mut ysafe, n, incy); 
            black_box ( srotm_safe ( 
                black_box(xcoral_aarch64), 
                black_box(ycoral_aarch64), 
                black_box(&param), 
            ))
        }); 
    });

    group.bench_function("srotm_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box ( srotm_neon ( 
                black_box(n), 
                black_box(&mut xneon), 
                black_box(incx), 
                black_box(&mut yneon), 
                black_box(incy), 
                black_box(&param), 
            )); 
        }); 
    });

    group.bench_function("srotm_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box ( cblas_srotm ( 
                black_box(n as i32), 
                black_box(xblas.as_mut_ptr()), 
                black_box(incx as i32), 
                black_box(yblas.as_mut_ptr()), 
                black_box(incy as i32), 
                black_box(param.as_ptr()), 
            )); 
        }); 
    });
}

pub fn srotm_strided(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 2;
    let incy = 3; 

    // flag = -1; full givens
    let param = [-1.0, 3.14, 3.14, 3.14, 3.14]; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 

    // look away before this blinds you
    let mut xsafe = xbuf.clone(); 
    let mut ysafe = ybuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut yneon = ybuf.clone(); 
    let mut xblas = xbuf.clone(); 
    let mut yblas = ybuf.clone(); 

    let mut group = c.benchmark_group("srotm_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 4))); 

    group.bench_function("srotm_coral", |b| { 
        b.iter(|| { 
            // includes creation time; neglible 
            let xcoral_aarch64 = make_view_mut(&mut xsafe, n, incx); 
            let ycoral_aarch64 = make_view_mut(&mut ysafe, n, incy); 
            black_box ( srotm_safe ( 
                black_box(xcoral_aarch64), 
                black_box(ycoral_aarch64), 
                black_box(&param), 
            ))
        }); 
    });

    group.bench_function("srotm_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box ( srotm_neon ( 
                black_box(n), 
                black_box(&mut xneon), 
                black_box(incx), 
                black_box(&mut yneon), 
                black_box(incy), 
                black_box(&param), 
            )); 
        }); 
    });

    group.bench_function("srotm_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box ( cblas_srotm ( 
                black_box(n as i32), 
                black_box(xblas.as_mut_ptr()), 
                black_box(incx as i32), 
                black_box(yblas.as_mut_ptr()), 
                black_box(incy as i32), 
                black_box(param.as_ptr()), 
            )); 
        }); 
    });
}

criterion_group!(benches, srotm_contiguous, srotm_strided); 
criterion_main!(benches); 
