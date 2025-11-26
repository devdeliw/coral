mod common; 
use common::{
    make_view_ref, 
    make_matview_mut, 
    bytes, 
    make_strided_mat, 
    make_strided_vec, 
};

use criterion::{ 
    criterion_main, 
    criterion_group, 
    Criterion, 
    Throughput, 
}; 

use blas_src as _; 
use cblas_sys::{cblas_sger, CBLAS_LAYOUT}; 
use coral_aarch64::level2::sger as sger_neon;
use coral::{level2::sger as sger_safe}; 

pub fn sger_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let m = 1024; 
    let lda = m; 

    let incx = 1;
    let incy = 1; 

    let alpha = 3.1415926; 
    
    let abuf = make_strided_mat(m, n, lda); 
    let xbuf = make_strided_vec(m, incx); 
    let ybuf = make_strided_vec(n, incy); 

    let mut asafe_buf = abuf.clone(); 
    let mut aneon_buf = abuf.clone(); 
    let mut ablas_buf = abuf.clone();

    let xsafe = make_view_ref(&xbuf, m, incx); 
    let ysafe = make_view_ref(&ybuf, n, incy); 

    let mut group = c.benchmark_group("sger_contiguous"); 
    group.throughput(Throughput::Bytes(bytes(m * n, 2)));

    group.bench_function("sger_coral", |b| { 
        b.iter(|| { 
            let asafe = make_matview_mut(&mut asafe_buf, m, n, lda); 
            sger_safe(alpha, asafe, xsafe, ysafe); 
        }); 
    });

    group.bench_function("sger_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            sger_neon ( 
                m, n, alpha, 
                &xbuf, incx, 
                &ybuf, incy, 
                &mut aneon_buf, lda
            )
        }); 
    }); 

    group.bench_function("sger_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_sger (
                CBLAS_LAYOUT::CblasColMajor, 
                m as i32,
                n as i32,
                alpha, 
                xbuf.as_ptr(), 
                incx as i32, 
                ybuf.as_ptr(),
                incy as i32, 
                ablas_buf.as_mut_ptr(),
                lda as i32, 
            )
        }); 
    }); 
}  

criterion_group!(benches, sger_contiguous); 
criterion_main!(benches); 
