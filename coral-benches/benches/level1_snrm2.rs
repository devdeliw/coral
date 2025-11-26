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
use cblas_sys::cblas_snrm2; 
use coral::level1::snrm2 as snrm2_safe; 
use coral_aarch64::level1::snrm2 as snrm2_neon; 

pub fn snrm2_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let inc = 1; 
    let xbuf = make_strided_vec(n, inc); 
    let xvec = make_view_ref(&xbuf, n, inc); 

    let mut group = c.benchmark_group("snrm2_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n, 1))); 

    group.bench_function("snrm2_coral", |b| { 
        b.iter(|| {
            black_box(snrm2_safe(black_box(xvec))); 
        });
    });

    group.bench_function("snrm2_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box(snrm2_neon(
                black_box(n), 
                black_box(&xbuf), 
                black_box(1)
            )); 
        });
    }); 

    group.bench_function("snrm2_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box(cblas_snrm2(
                black_box(n as i32),
                black_box(xbuf.as_ptr()),
                black_box(1),
            )); 
        });
    });
}

pub fn snrm2_strided(c: &mut Criterion) { 
    let n = 1000000; 
    let inc = 2;
    let xbuf = make_strided_vec(n, inc); 
    let xvec = make_view_ref(&xbuf, n, inc); 

    let mut group = c.benchmark_group("snrm2_strided"); 
    group.throughput(Throughput::Bytes(bytes(n, 1))); 

    group.bench_function("snrm2_coral", |b| { 
        b.iter(|| {
            black_box(snrm2_safe(black_box(xvec))); 
        });
    });

    group.bench_function("snrm2_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            black_box(snrm2_neon(
                black_box(n), 
                black_box(&xbuf), 
                black_box(inc)
            )); 
        });
    }); 

    group.bench_function("snrm2_cblas", |b| { 
        b.iter(|| unsafe { 
            black_box(cblas_snrm2(
                black_box(n as i32),
                black_box(xbuf.as_ptr()),
                black_box(inc as i32),
            )); 
        });
    });
}

criterion_group!(benches, snrm2_contiguous, snrm2_strided);
criterion_main!(benches);




