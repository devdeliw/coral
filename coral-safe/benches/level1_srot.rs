//! `VectorMut` creation times are included 
//! in the timed-part of the `safe` benchmark. 
//! This is because the `VectorMut`s are <'a, T> 
//! with `self.data<&'a mut [f32]>. Consequently 
//! they can not be pre-allocated in a closure.
//!
//! This is almost negligible though. Additionally, 
//! buffers aren't refreshed after each iteration. 
//! However, everything is [f32]; same amount of work. 

mod common; 
use common::{make_strided_vec, bytes, make_view_mut}; 

use criterion::{ 
    criterion_main, 
    criterion_group, 
    Criterion,
    Throughput, 
    black_box
};

use blas_src as _; 
use cblas_sys::{cblas_srot, cblas_srotm};
use coral_safe::level1::{
    srot as srot_safe, 
    srotm as srotm_safe, 
}; 
use coral::level1::{
    srot as srot_neon, 
    srotm as srotm_neon,
}; 

pub fn srot_contiguous(c: &mut Criterion) { 
    let n = 1000000;
    let incx = 1; 
    let incy = 1; 

    let c_val = 0.8; 
    let s_val = 0.6; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 

    // look away before this blinds you 
    let mut xsafe = xbuf.clone(); 
    let mut ysafe = ybuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut yneon = ybuf.clone(); 
    let mut xblas = xbuf.clone(); 
    let mut yblas = ybuf.clone();  

    let mut group = c.benchmark_group("srot_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n, 4))); 

    group.bench_function("srot_coral_safe", |b| { 
        b.iter(|| {
            // unavoidable egligible creation time
            let xcoral = make_view_mut(&mut xsafe, n, incx); 
            let ycoral = make_view_mut(&mut ysafe, n, incy); 

            black_box( srot_safe (
                black_box(xcoral), 
                black_box(ycoral), 
                black_box(c_val), 
                black_box(s_val), 
            )); 
        }); 
    }); 

    group.bench_function("srot_coral_neon", |b| { 
        b.iter(|| { 
            black_box ( srot_neon ( 
                black_box(n), 
                black_box(&mut xneon), 
                black_box(incx), 
                black_box(&mut yneon), 
                black_box(incy), 
                black_box(c_val), 
                black_box(s_val), 
            ));
        });
    }); 

    group.bench_function("srot_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box( cblas_srot ( 
                black_box(n as i32),
                black_box(xblas.as_mut_ptr()), 
                black_box(incx as i32), 
                black_box(yblas.as_mut_ptr()), 
                black_box(incy as i32), 
                black_box(c_val), 
                black_box(s_val),
            )); 
        });
    }); 
}

pub fn srot_strided(c: &mut Criterion) { 
    let n = 1000000;
    let incx = 2; 
    let incy = 3; 

    let c_val = 0.8; 
    let s_val = 0.6; 

    let xbuf = make_strided_vec(n, incx); 
    let ybuf = make_strided_vec(n, incy); 

    let mut xsafe = xbuf.clone(); 
    let mut ysafe = ybuf.clone(); 
    let mut xneon = xbuf.clone(); 
    let mut yneon = ybuf.clone(); 
    let mut xblas = xbuf.clone(); 
    let mut yblas = ybuf.clone();  

    let mut group = c.benchmark_group("srot_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 4))); 

    group.bench_function("srot_coral_safe", |b| { 
        b.iter(|| {
            // unavoidable egligible creation time
            let xcoral = make_view_mut(&mut xsafe, n, incx); 
            let ycoral = make_view_mut(&mut ysafe, n, incy); 

            black_box( srot_safe (
                black_box(xcoral), 
                black_box(ycoral), 
                black_box(c_val), 
                black_box(s_val), 
            )); 
        }); 
    }); 

    group.bench_function("srot_coral_neon", |b| { 
        b.iter(|| { 
            black_box ( srot_neon ( 
                black_box(n), 
                black_box(&mut xneon), 
                black_box(incx), 
                black_box(&mut yneon), 
                black_box(incy), 
                black_box(c_val), 
                black_box(s_val), 
            ));
        });
    }); 

    group.bench_function("srot_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box( cblas_srot ( 
                black_box(n as i32),
                black_box(xblas.as_mut_ptr()), 
                black_box(incx as i32), 
                black_box(yblas.as_mut_ptr()), 
                black_box(incy as i32), 
                black_box(c_val), 
                black_box(s_val),
            )); 
        });
    }); 
}

criterion_group!(benches, srot_contiguous, srot_strided); 
criterion_main!(benches);
