//! `VectorMut` creation times are included 
//! in the timed-part of the `safe` benchmark. 
//! This is because the `VectorMut`s are <'a, T> 
//! with `self.data<&'a mut [f32]>. Consequently 
//! they can not be pre-allocated in a closure.
//!
//! This is almost negligible though. The `safe` 
//! benchmark still outperforms over the `unsafe` NEON 
//! implementation for contiguous buffers.  

mod common; 
use common::{make_strided_vec, bytes, make_view_mut}; 

use criterion::{ 
    criterion_group, 
    criterion_main,
    Criterion, 
    Throughput, 
    black_box, 
};

use blas_src as _; 
use cblas_sys::cblas_sswap; 
use coral_safe::level1::sswap as sswap_safe; 
use coral::level1::sswap as sswap_neon; 

pub fn sswap_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 1; 
    let incy = 1;
    let mut xsafe = make_strided_vec(n, incx); 
    let mut ysafe = make_strided_vec(n, incy); 

    let mut xneon = xsafe.clone(); 
    let mut yneon = ysafe.clone(); 

    let mut xblas = xsafe.clone(); 
    let mut yblas = ysafe.clone();

    let mut group = c.benchmark_group("sswap_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n, 4)));

    group.bench_function("sswap_coral_safe", |b| { 
        b.iter(|| {
            // creation time is included
            // unavoidable due to VectorMut<'_, f32> 
            // owning a &'a mut [f32]
            //
            // basically negligible though
            let xcoral = make_view_mut(&mut xsafe, n, incx); 
            let ycoral = make_view_mut(&mut ysafe, n, incy); 
            black_box(sswap_safe(black_box(xcoral), black_box(ycoral))); 
        });
    });

    group.bench_function("sswap_coral_neon", |b| { 
        b.iter(|| { 
            black_box( sswap_neon ( 
                black_box(n), 
                black_box(&mut xneon), 
                black_box(incx), 
                black_box(&mut yneon), 
                black_box(incy)
            )); 
        }); 
    }); 

    group.bench_function("sswap_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box( cblas_sswap ( 
                black_box(n as i32), 
                black_box(xblas.as_mut_ptr()), 
                black_box(incx as i32), 
                black_box(yblas.as_mut_ptr()), 
                black_box(incy as i32)
            )); 
        }); 
    }); 
}

pub fn sswap_strided(c: &mut Criterion) { 
    let n = 1000000; 
    let incx = 2; 
    let incy = 3;
    let mut xsafe = make_strided_vec(n, incx); 
    let mut ysafe = make_strided_vec(n, incy); 

    let mut xneon = xsafe.clone(); 
    let mut yneon = ysafe.clone(); 

    let mut xblas = xsafe.clone(); 
    let mut yblas = ysafe.clone();

    let mut group = c.benchmark_group("sswap_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 4)));

    group.bench_function("sswap_coral_safe", |b| { 
        b.iter(|| {
            let xcoral = make_view_mut(&mut xsafe, n, incx); 
            let ycoral = make_view_mut(&mut ysafe, n, incy); 
            black_box(sswap_safe(black_box(xcoral), black_box(ycoral))); 
        }); 
    }); 

    group.bench_function("sswap_coral_neon", |b| { 
        b.iter(|| { 
            black_box( sswap_neon ( 
                black_box(n), 
                black_box(&mut xneon), 
                black_box(incx), 
                black_box(&mut yneon), 
                black_box(incy)
            )); 
        }); 
    }); 

    group.bench_function("sswap_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box( cblas_sswap ( 
                black_box(n as i32), 
                black_box(xblas.as_mut_ptr()), 
                black_box(incx as i32), 
                black_box(yblas.as_mut_ptr()), 
                black_box(incy as i32)
            )); 
        }); 
    }); 
}

criterion_group!(benches, sswap_contiguous, sswap_strided);
criterion_main!(benches);

