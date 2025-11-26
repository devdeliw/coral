mod common;
use common::{
    make_view_ref,
    make_matview_mut,
    bytes,
    make_strided_mat,
    make_strided_vec,
    SIZES,
};

use criterion::{
    criterion_main,
    criterion_group,
    BenchmarkId,
    Criterion,
    Throughput,
};

use blas_src as _;
use cblas_sys::{cblas_sger, CBLAS_LAYOUT};
use coral_aarch64::level2::sger as sger_neon;
use coral::level2::sger as sger_safe;

pub fn sger_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sger_contiguous_sweep");

    for &n in SIZES {
        let m = n;
        let lda = m;
        let incx = 1;
        let incy = 1;
        let alpha: f32 = 3.1415926;

        group.throughput(Throughput::Bytes(bytes(m * n, 2)));

        group.bench_with_input(
            BenchmarkId::new("sger_coral", n),
            &n,
            |b, &_n| {
                let mut asafe_buf = make_strided_mat(m, n, lda);
                let xbuf = make_strided_vec(m, incx);
                let ybuf = make_strided_vec(n, incy);
                let xsafe = make_view_ref(&xbuf, m, incx);
                let ysafe = make_view_ref(&ybuf, n, incy);

                b.iter(|| {
                    let asafe = make_matview_mut(&mut asafe_buf, m, n, lda);
                    sger_safe(alpha, asafe, xsafe, ysafe);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sger_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                let mut aneon_buf = make_strided_mat(m, n, lda);
                let xbuf = make_strided_vec(m, incx);
                let ybuf = make_strided_vec(n, incy);

                b.iter(|| {
                    sger_neon(
                        m,
                        n,
                        alpha,
                        &xbuf,
                        incx,
                        &ybuf,
                        incy,
                        &mut aneon_buf,
                        lda,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sger_cblas", n),
            &n,
            |b, &_n| {
                let mut ablas_buf = make_strided_mat(m, n, lda);
                let xbuf = make_strided_vec(m, incx);
                let ybuf = make_strided_vec(n, incy);

                b.iter(|| unsafe {
                    cblas_sger(
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
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, sger_contiguous_sweep);
criterion_main!(benches);
