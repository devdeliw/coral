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
use cblas_sys::cblas_isamax; 
use coral_safe::types::VectorRef; 
use coral_safe::level1::isamax as isamax_safe; 
use coral::level1::isamax as isamax_neon;

#[inline]
fn bytes(n: usize) -> u64 { 
    (n * std::mem::size_of::<f32>()) as u64
}

#[inline] 
fn make_view(x: &[f32]) -> VectorRef<'_, f32> { 
    VectorRef::new(x, x.len(), 1, 0).expect("x view")
}

pub fn isamax_contiguous(c: &mut Criterion) { 
    let n = 1000000; 
    let inc = 1; 
    let xbuf = make_strided_vec(n, inc); 
    let xvec = make_view(&xbuf); 

    let mut group = c.benchmark_group("isamax_contiguous");
    group.throughput(Throughput::Bytes(bytes(n))); 

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
    let xvec = make_view(&xbuf); 

    let mut group = c.benchmark_group("isamax_strided"); 
    group.throughput(Throughput::Bytes(bytes(n))); 

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

criterion_group!(benches, isamax_contiguous, isamax_strided);
criterion_main!(benches);

