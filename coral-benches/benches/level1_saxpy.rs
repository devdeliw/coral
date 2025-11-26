mod common; 
use common::{make_strided_vec, bytes, make_view_ref, make_view_mut}; 

use criterion::{ 
    criterion_group, 
    criterion_main, 
    Criterion, 
    Throughput,
    black_box
}; 

use blas_src as _; 
use cblas_sys::cblas_saxpy; 
use coral::level1::saxpy as saxpy_safe; 
use coral_aarch64::level1::saxpy as saxpy_neon; 

pub fn saxpy_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 1; 
    let incy = 1; 
    let alpha = 3.1415926535; 

    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 
    
    let xcoral_aarch64 = make_view_ref(&xbuf, n, incx);

    let mut group = c.benchmark_group("saxpy_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n, 3))); 

    group.bench_function("saxpy_coral", |b| { 
        b.iter(|| { 
            let ycoral_aarch64 = make_view_mut(&mut ybuf, n, incy); 
            black_box(saxpy_safe(alpha, black_box(xcoral_aarch64), black_box(ycoral_aarch64))); 
        }); 
    });

    group.bench_function("saxpy_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box ( saxpy_neon ( 
                black_box(n), 
                black_box(alpha), 
                black_box(&xbuf), 
                black_box(incx), 
                black_box(&mut ybuf), 
                black_box(incy)
      )); 
        }); 
    });

    group.bench_function("saxpy_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box ( cblas_saxpy ( 
                black_box(n as i32), 
                black_box(alpha), 
                black_box(xbuf.as_ptr()), 
                black_box(incx as i32), 
                black_box(ybuf.as_mut_ptr()), 
                black_box(incy as i32), 
            )); 
        }); 
    }); 
}

pub fn saxpy_strided(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 2; 
    let incy = 3; 
    let alpha = 3.1415926535; 

    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let xcoral_aarch64 = make_view_ref(&xbuf, n, incx);

    let mut group = c.benchmark_group("saxpy_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 3))); 

    group.bench_function("saxpy_coral", |b| { 
        b.iter(|| { 
            let ycoral_aarch64 = make_view_mut(&mut ybuf, n, incy); 
            black_box(saxpy_safe(alpha, black_box(xcoral_aarch64), black_box(ycoral_aarch64))); 
        }); 
    });

    group.bench_function("saxpy_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box ( saxpy_neon ( 
                black_box(n), 
                black_box(alpha), 
                black_box(&xbuf), 
                black_box(incx), 
                black_box(&mut ybuf), 
                black_box(incy)
            )); 
        }); 
    });

    group.bench_function("saxpy_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box ( cblas_saxpy ( 
                black_box(n as i32), 
                black_box(alpha), 
                black_box(xbuf.as_ptr()), 
                black_box(incx as i32), 
                black_box(ybuf.as_mut_ptr()), 
                black_box(incy as i32), 
            )); 
        }); 
    }); 
}

criterion_group!(benches, saxpy_contiguous, saxpy_strided);
criterion_main!(benches);

