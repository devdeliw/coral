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
use cblas_sys::cblas_isamax; 
use coral_safe::types::VectorRef; 
use coral_safe::level1::isamax as isamax_safe; 
use coral::level1::isamax as isamax_neon;


pub fn isamax_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let inc = 1; 
    let xbuf = make_strided_vec(n, inc); 
    let xvec = make_view_ref(&xbuf, n, inc); 

    let mut group = c.benchmark_group("isamax_contiguous");
    group.throughput(Throughput::Bytes(bytes(n, 1))); 

    group.bench_function("isamax_coral_safe", |b| { 
        b.iter(|| {
            black_box(isamax_safe(black_box(xvec))); 
        });
    });

    group.bench_function("isamax_coral_neon", |b| { 
        b.iter(|| { 
            black_box(isamax_neon(
                black_box(n), 
                black_box(&xbuf), 
                black_box(1)
            )); 
        });
    }); 

    group.bench_function("isamax_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box(cblas_isamax(
                black_box(n as i32),
                black_box(xbuf.as_ptr()),
                black_box(1),
            )); 
        });
    });
}

pub fn isamax_strided(c: &mut Criterion) { 
    let n = 1000000; 
    let inc = 2;
    let xbuf = make_strided_vec(n, inc); 
    let xvec = make_view_ref(&xbuf, n, inc); 

    let mut group = c.benchmark_group("isamax_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 1))); 

    group.bench_function("isamax_coral_safe", |b| { 
        b.iter(|| {
            black_box(isamax_safe(black_box(xvec))); 
        });
    });

    group.bench_function("isamax_coral_neon", |b| { 
        b.iter(|| { 
            black_box(isamax_neon(
                black_box(n), 
                black_box(&xbuf), 
                black_box(inc)
            )); 
        });
    }); 

    group.bench_function("isamax_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box(cblas_isamax(
                black_box(n as i32),
                black_box(xbuf.as_ptr()),
                black_box(inc as i32),
            )); 
        });
    });
}

criterion_group!(benches, isamax_contiguous, isamax_strided);
criterion_main!(benches);

