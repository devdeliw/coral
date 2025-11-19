//! This bench is not compared to any blas or 
//! neon aarch64 implementations. 
//!
//! It is just for me to see if I can make 
//! a safe implementation of a mini GEMV 
//! reach max thrpt. 

mod common; 
use common::{
    make_strided_mat,
    make_strided_vec, 
    make_view_ref, 
    make_view_mut,
    make_matview_ref, 
    bytes
}; 

use criterion::{
    criterion_main, 
    criterion_group, 
    Criterion,
    Throughput, 
    black_box
};

use coral_safe::fused::saxpyf::saxpyf; 

pub fn saxpyf_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let incx = 1; 
    let incy = 1; 
    let lda = n; 

    let abuf = make_strided_mat(n, n, lda); 
    let xbuf = make_strided_vec(n, incx); 
    let mut ybuf = make_strided_vec(n, incy); 

    let xcoral = make_view_ref(&xbuf, n, incx); 
    let acoral = make_matview_ref(&abuf, n, n, lda);
    
    let mut group = c.benchmark_group("saxpyf_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(n*n + 3*n, 1))); 

    group.bench_function("saxpyf_coral_safe", |b| { 
        b.iter(|| { 
            let ycoral = make_view_mut(&mut ybuf, n, incy); 
            
            saxpyf(
                black_box(acoral), 
                black_box(xcoral),
                black_box(ycoral)
            ); 
        }); 
    }); 

}

criterion_group!(benches, saxpyf_contiguous); 
criterion_main!(benches); 
