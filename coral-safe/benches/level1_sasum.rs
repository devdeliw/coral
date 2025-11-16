mod common;
use common::make_strided_vec; 

use criterion::{
    criterion_group, 
    criterion_main,
    Criterion,
    Throughput, 
    black_box,
}; 

use blas_src as _; 
use cblas_sys::cblas_sasum; 
use coral_safe::types::VectorRef; 
use coral_safe::level1::sasum as sasum_safe; 
use coral::level1::sasum as sasum_neon;

#[inline]
fn bytes(n: usize) -> u64 { 
    (n * std::mem::size_of::<f32>()) as u64
}

#[inline] 
fn make_view(x: &[f32]) -> VectorRef<'_, f32> { 
    VectorRef::new(x, x.len(), 1, 0).expect("x view")
}

pub fn sasum_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let inc = 1; 
    let xbuf = make_strided_vec(n, inc); 
    let xvec = make_view(&xbuf); 

    let mut group = c.benchmark_group("sasum_contiguous");
    group.throughput(Throughput::Bytes(bytes(n))); 

    group.bench_function("sasum_coral_safe", |b| { 
        b.iter(|| {
            black_box(sasum_safe(black_box(xvec))); 
        });
    });

    group.bench_function("sasum_coral_neon", |b| { 
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
    let xvec = make_view(&xbuf); 

    let mut group = c.benchmark_group("sasum_strided"); 
    group.throughput(Throughput::Bytes(bytes(n))); 

    group.bench_function("sasum_coral_safe", |b| { 
        b.iter(|| {
            black_box(sasum_safe(black_box(xvec))); 
        });
    });

    group.bench_function("sasum_coral_neon", |b| { 
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

criterion_group!(benches, sasum_contiguous, sasum_strided);
criterion_main!(benches);
