mod common; 
use common::{
    make_strided_mat, 
    make_matview_ref, 
    make_matview_mut, 
}; 

use criterion::{
    criterion_main, 
    criterion_group, 
    Criterion, 
}; 

use blas_src as _; 
use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE}; 
use coral_aarch64::level3::sgemm as sgemm_neon; 
use coral_aarch64::enums::CoralTranspose as NeonTranspose; 
use coral::level3::sgemm as sgemm_safe; 
use coral::types::CoralTranspose; 

use faer::{mat, Parallelism}; 
use faer::linalg::matmul::matmul as sgemm_faer; 

#[inline(always)]
fn faer_ref<'a>(
    ptr: *const f32, 
    m  : usize,
    n  : usize,
    ld : usize
) -> faer::MatRef<'a, f32> {
    unsafe {
        mat::from_raw_parts::<f32>(ptr, m, n, 1, ld as isize)
    } 
}
#[inline(always)]
fn faer_mut<'a>(
    ptr: *mut f32,
    m  : usize, 
    n  : usize, 
    ld : usize
) -> faer::MatMut<'a, f32> {
    unsafe {
        mat::from_raw_parts_mut::<f32>(ptr, m, n, 1, ld as isize)
    } 
}


pub fn sgemm_nn_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let alpha = 3.1415926; 
    let beta = 2.71828; 

    let opa = CoralTranspose::NoTrans; 
    let opb = CoralTranspose::NoTrans; 

    let lda = n; 
    let ldb = n; 
    let ldc = n;

    let abuf = make_strided_mat(n, n, lda); 
    let bbuf = make_strided_mat(n, n, ldb); 
    let cbuf = make_strided_mat(n, n, ldc);

    let mut csafe = cbuf.clone(); 
    let mut cneon = cbuf.clone(); 
    let mut cblas = cbuf.clone(); 

    let afaer = faer_ref(abuf.as_ptr(), n, n, lda); 
    let bfaer = faer_ref(bbuf.as_ptr(), n, n, lda); 
    let mut cfaer = cbuf.clone(); 

    let acoral_aarch64 = make_matview_ref(&abuf, n, n, lda);
    let bcoral_aarch64 = make_matview_ref(&bbuf, n, n, ldb); 

    let mut group = c.benchmark_group("sgemm_nn_contiguous"); 
    group.throughput(criterion::Throughput::Elements((2 * n * n * n) as u64));

    group.bench_function("sgemm_nn_coral", |b| { 
        b.iter(|| {     
            let ccoral_aarch64 = make_matview_mut(&mut csafe, n, n, ldc);
            sgemm_safe(opa, opb, alpha, beta, acoral_aarch64, bcoral_aarch64, ccoral_aarch64); 
        })
    });

    group.bench_function("sgemm_nn_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            sgemm_neon ( 
                NeonTranspose::NoTranspose, 
                NeonTranspose::NoTranspose, 
                n, n, n, 
                alpha, 
                abuf.as_ptr(),
                lda, 
                bbuf.as_ptr(), 
                ldb, 
                beta, 
                cneon.as_mut_ptr(), 
                ldc, 
            );
        })
    }); 

    group.bench_function("sgemm_nn_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_sgemm ( 
                CBLAS_LAYOUT::CblasColMajor, 
                CBLAS_TRANSPOSE::CblasNoTrans, 
                CBLAS_TRANSPOSE::CblasNoTrans, 
                n as i32, n as i32, n as i32, 
                alpha, 
                abuf.as_ptr(), 
                lda as i32, 
                bbuf.as_ptr(), 
                ldb as i32, 
                beta, 
                cblas.as_mut_ptr(), 
                ldc as i32, 
            )
        }); 
    });

    group.bench_function("sgemm_nn_faer", |b| { 
        b.iter(|| { 
            let mut c_mut = faer_mut(cfaer.as_mut_ptr(), n, n, ldc); 

            sgemm_faer ( 
                c_mut.as_mut(), 
                afaer, 
                bfaer, 
                Some(beta), 
                alpha, 
                Parallelism::None, 
            )
        }); 
    });
}

pub fn sgemm_tt_contiguous(c: &mut Criterion) { 
    let n = 1024; 
    let alpha = 3.1415926; 
    let beta = 2.71828; 

    let opa = CoralTranspose::Trans; 
    let opb = CoralTranspose::Trans; 

    let lda = n; 
    let ldb = n; 
    let ldc = n;

    let abuf = make_strided_mat(n, n, lda); 
    let bbuf = make_strided_mat(n, n, ldb); 
    let cbuf = make_strided_mat(n, n, ldc);

    let mut csafe = cbuf.clone(); 
    let mut cneon = cbuf.clone(); 
    let mut cblas = cbuf.clone(); 

    let acoral_aarch64 = make_matview_ref(&abuf, n, n, lda);
    let bcoral_aarch64 = make_matview_ref(&bbuf, n, n, ldb); 

    let mut group = c.benchmark_group("sgemm_tt_contiguous"); 
    group.throughput(criterion::Throughput::Elements((2 * n * n * n) as u64));

    group.bench_function("sgemm_tt_coral", |b| { 
        b.iter(|| {     
            let ccoral_aarch64 = make_matview_mut(&mut csafe, n, n, ldc);
            sgemm_safe(opa, opb, alpha, beta, acoral_aarch64, bcoral_aarch64, ccoral_aarch64); 
        })
    });

    group.bench_function("sgemm_tt_coral_aarch64_neon", |b| { 
        b.iter(|| { 
            sgemm_neon ( 
                NeonTranspose::Transpose, 
                NeonTranspose::Transpose, 
                n, n, n, 
                alpha, 
                abuf.as_ptr(),
                lda, 
                bbuf.as_ptr(), 
                ldb, 
                beta, 
                cneon.as_mut_ptr(), 
                ldc, 
            );
        })
    }); 

    group.bench_function("sgemm_tt_cblas", |b| { 
        b.iter(|| unsafe { 
            cblas_sgemm ( 
                CBLAS_LAYOUT::CblasColMajor, 
                CBLAS_TRANSPOSE::CblasTrans, 
                CBLAS_TRANSPOSE::CblasTrans, 
                n as i32, n as i32, n as i32, 
                alpha, 
                abuf.as_ptr(), 
                lda as i32, 
                bbuf.as_ptr(), 
                ldb as i32, 
                beta, 
                cblas.as_mut_ptr(), 
                ldc as i32, 
            )
        }); 
    }); 
}

criterion_group!(benches, sgemm_nn_contiguous, sgemm_tt_contiguous); 
criterion_main!(benches);

