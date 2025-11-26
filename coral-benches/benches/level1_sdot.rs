mod common; 
use common::{make_strided_vec, bytes, make_view_ref}; 

use criterion::{ 
    criterion_group, 
    criterion_main, 
    Criterion, 
    Throughput, 
    black_box, 
}; 

use blas_src as _; 
use cblas_sys::cblas_sdot; 
use coral::level1::sdot as sdot_safe; 
use coral_aarch64::level1::sdot as sdot_neon;

pub fn sdot_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 1; 
    let incy = 1; 
    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 

    let xvec = make_view_ref(&xbuf, n, incx); 
    let yvec = make_view_ref(&ybuf, n, incy); 

    let mut group = c.benchmark_group("sdot_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n, 2))); 

    group.bench_function("sdot_coral", |b| { 
        b.iter(|| { 
            black_box(sdot_safe(black_box(xvec), black_box(yvec))); 
        }); 
    });

    group.bench_function("sdot_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box ( sdot_neon ( 
                black_box(n), 
                black_box(&xbuf),
                black_box(incx), 
                black_box(&ybuf),
                black_box(incy)
            )); 
        }); 
    });

    group.bench_function("sdot_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box ( cblas_sdot ( 
                black_box(n as i32), 
                black_box(xbuf.as_ptr()),
                black_box(incx as i32), 
                black_box(ybuf.as_ptr()),
                black_box(incy as i32)
            )); 
        }); 
    }); 
}

pub fn sdot_strided(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 2; 
    let incy = 3; 
    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 

    let xvec = make_view_ref(&xbuf, n, incx); 
    let yvec = make_view_ref(&ybuf, n, incy); 

    let mut group = c.benchmark_group("sdot_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 2))); 

    group.bench_function("sdot_coral", |b| { 
        b.iter(|| { 
            black_box(sdot_safe(black_box(xvec), black_box(yvec))); 
        }); 
    });

    group.bench_function("sdot_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box ( sdot_neon ( 
                black_box(n), 
                black_box(&xbuf),
                black_box(incx), 
                black_box(&ybuf),
                black_box(incy)
            )); 
        }); 
    });

    group.bench_function("sdot_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box ( cblas_sdot ( 
                black_box(n as i32), 
                black_box(xbuf.as_ptr()),
                black_box(incx as i32), 
                black_box(ybuf.as_ptr()),
                black_box(incy as i32)
            )); 
        }); 
    }); 
}

criterion_group!(benches, sdot_contiguous, sdot_strided);
criterion_main!(benches);


