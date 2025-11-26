mod common;
use common::{make_strided_vec, bytes, make_view_ref, SIZES};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
    black_box,
};

use blas_src as _;
use cblas_sys::cblas_snrm2;
use coral::level1::snrm2 as snrm2_safe;
use coral_aarch64::level1::snrm2 as snrm2_neon;

pub fn snrm2_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("snrm2_contiguous_sweep");

    for &n in SIZES {
        let inc = 1;
        let xbuf: &'static [f32] = Box::leak(make_strided_vec(n, inc).into_boxed_slice());
        let xvec = make_view_ref(xbuf, n, inc);

        group.throughput(Throughput::Bytes(bytes(n, 1)));

        group.bench_with_input(
            BenchmarkId::new("snrm2_coral", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    black_box(snrm2_safe(black_box(xvec)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("snrm2_coral_aarch64_neon", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    black_box(snrm2_neon(
                        black_box(n),
                        black_box(xbuf),
                        black_box(inc),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("snrm2_cblas", n),
            &n,
            |b, &_n| {
                b.iter(|| unsafe {
                    black_box(cblas_snrm2(
                        black_box(n as i32),
                        black_box(xbuf.as_ptr()),
                        black_box(inc as i32),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, snrm2_contiguous_sweep);
criterion_main!(benches);
