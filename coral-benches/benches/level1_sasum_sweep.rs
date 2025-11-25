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
use cblas_sys::cblas_sasum;
use coral_safe::level1::sasum as sasum_safe;
use coral::level1::sasum as sasum_neon;

pub fn sasum_contiguous_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sasum_contiguous_sweep");

    for &n in SIZES {
        let inc = 1;
        let xbuf: &'static [f32] = Box::leak(make_strided_vec(n, inc).into_boxed_slice());
        let xvec = make_view_ref(xbuf, n, inc);

        group.throughput(Throughput::Bytes(bytes(n, 1)));

        group.bench_with_input(
            BenchmarkId::new("sasum_coral_safe", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    black_box(sasum_safe(black_box(xvec)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sasum_coral_neon", n),
            &n,
            |b, &_n| {
                b.iter(|| {
                    black_box(sasum_neon(
                        black_box(n),
                        black_box(xbuf),
                        black_box(inc),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sasum_cblas", n),
            &n,
            |b, &_n| {
                b.iter(|| unsafe {
                    black_box(cblas_sasum(
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

criterion_group!(benches, sasum_contiguous_sweep);
criterion_main!(benches);
