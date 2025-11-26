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
use cblas_sys::cblas_sasum; 
use coral::level1::sasum as sasum_safe; 
use coral_aarch64::level1::sasum as sasum_neon;

pub fn sasum_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let inc = 1; 
    let xbuf = make_strided_vec(n, inc); 
    let xvec = make_view_ref(&xbuf, n, inc); 

    let mut group = c.benchmark_group("sasum_contiguous");
    group.throughput(Throughput::Bytes(bytes(n, 1))); 

    group.bench_function("sasum_coral", |b| { 
        b.iter(|| {
            black_box(sasum_safe(black_box(xvec))); 
        });
    });

    group.bench_function("sasum_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box(sasum_neon(
                black_box(n), 
                black_box(&xbuf), 
                black_box(1)
            )); 
        });
    }); 

    group.bench_function("sasum_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box(cblas_sasum(
                black_box(n as i32),
                black_box(xbuf.as_ptr()),
                black_box(1),
            )); 
        });
    });
}

pub fn sasum_strided(c: &mut Criterion) { 
    let n = 1000000; 
    let inc = 2;
    let xbuf = make_strided_vec(n, inc); 
    let xvec = make_view_ref(&xbuf, n, inc); 

    let mut group = c.benchmark_group("sasum_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 1))); 

    group.bench_function("sasum_coral", |b| { 
        b.iter(|| {
            black_box(sasum_safe(black_box(xvec))); 
        });
    });

    group.bench_function("sasum_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box(sasum_neon(
                black_box(n), 
                black_box(&xbuf), 
                black_box(inc)
            )); 
        });
    }); 

    group.bench_function("sasum_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box(cblas_sasum(
                black_box(n as i32),
                black_box(xbuf.as_ptr()),
                black_box(inc as i32),
            )); 
        });
    });
}

criterion_group!(benches, sasum_contiguous, sasum_strided);
criterion_main!(benches);
