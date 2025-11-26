mod common; 
use common::{make_strided_vec, bytes, make_view_ref, make_view_mut}; 

use criterion::{
    criterion_main, 
    criterion_group, 
    Criterion, 
    Throughput, 
    black_box,
};

use blas_src as _; 
use cblas_sys::cblas_scopy; 
use coral::level1::scopy as scopy_safe; 
use coral_aarch64::level1::scopy as scopy_neon; 

pub fn scopy_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 1; 
    let incy = 1; 
    
    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 

    let xcoral_aarch64 = make_view_ref(&xbuf, n, incx); 
    let mut ysafe = ybuf.clone(); 
    let mut yneon = ybuf.clone(); 
    let mut yblas = ybuf.clone(); 

    let mut group = c.benchmark_group("scopy_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n, 2))); 

    group.bench_function("scopy_coral", |b| { 
        b.iter(|| { 
            let ycoral_aarch64 = make_view_mut(&mut ysafe, n, incy); 
            black_box(scopy_safe(black_box(xcoral_aarch64), black_box(ycoral_aarch64))); 
        }); 
    }); 

    group.bench_function("scopy_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box ( scopy_neon ( 
                black_box(n), 
                black_box(&xbuf), 
                black_box(incx), 
                black_box(&mut yneon), 
                black_box(incy)
            ));
        }); 
    });

    group.bench_function("scopy_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box ( cblas_scopy ( 
                black_box(n as i32), 
                black_box(xbuf.as_ptr()), 
                black_box(incx as i32), 
                black_box(yblas.as_mut_ptr()), 
                black_box(incy as i32), 
            )); 
        }); 
    }); 
}

pub fn scopy_strided(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 2; 
    let incy = 3; 
    
    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 

    let xcoral_aarch64 = make_view_ref(&xbuf, n, incx); 
    let mut ysafe = ybuf.clone(); 
    let mut yneon = ybuf.clone(); 
    let mut yblas = ybuf.clone(); 

    let mut group = c.benchmark_group("scopy_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 2))); 

    group.bench_function("scopy_coral", |b| { 
        b.iter(|| { 
            let ycoral_aarch64 = make_view_mut(&mut ysafe, n, incy); 
            black_box(scopy_safe(black_box(xcoral_aarch64), black_box(ycoral_aarch64))); 
        }); 
    }); 

    group.bench_function("scopy_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box ( scopy_neon ( 
                black_box(n), 
                black_box(&xbuf), 
                black_box(incx), 
                black_box(&mut yneon), 
                black_box(incy)
            ));
        }); 
    });

    group.bench_function("scopy_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box ( cblas_scopy ( 
                black_box(n as i32), 
                black_box(xbuf.as_ptr()), 
                black_box(incx as i32), 
                black_box(yblas.as_mut_ptr()), 
                black_box(incy as i32), 
            )); 
        }); 
    }); 
}

criterion_group!(benches, scopy_contiguous, scopy_strided); 
criterion_main!(benches);

