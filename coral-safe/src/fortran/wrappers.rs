use std::slice;
use crate::types::{VectorRef, VectorMut, MatrixRef, MatrixMut};

pub unsafe fn ptr_to_view<'a>(
    n: i32,
    x: *const f32,
    incx: i32,
) -> VectorRef<'a, f32> { unsafe { 
    assert!(n >= 0, "n must be non-negative");
    if n == 0 {
        let xbuf: &'a [f32] = slice::from_raw_parts(x, 0);
        return VectorRef::new(xbuf, 0, 1, 0)
            .expect("VectorRef::new failed for n=0");
    }

    assert!(incx > 0, "incx must be positive"); 
    let n_usize   = n as usize;
    let incx_u    = incx as usize;
    let len       = 1 + (n_usize - 1) * incx_u;

    let xbuf: &'a [f32] = slice::from_raw_parts(x, len);

    VectorRef::new(xbuf, n_usize, incx_u, 0)
        .expect("x view failed")
}}

pub unsafe fn ptr_to_view_mut<'a>(
    n: i32,
    x: *mut f32,
    incx: i32,
) -> VectorMut<'a, f32> { unsafe { 
    assert!(n >= 0, "n must be non-negative");
    if n == 0 {
        let xbuf: &'a mut [f32] = slice::from_raw_parts_mut(x, 0);
        return VectorMut::new(xbuf, 0, 1, 0)
            .expect("VectorMut::new failed for n=0");
    }

    assert!(incx > 0, "incx must be positive");
    let n_usize   = n as usize;
    let incx_u    = incx as usize;
    let len       = 1 + (n_usize - 1) * incx_u;

    let xbuf: &'a mut [f32] = slice::from_raw_parts_mut(x, len);

    VectorMut::new(xbuf, n_usize, incx_u, 0)
        .expect("x mut view failed")
}}

pub unsafe fn ptr_to_mat_ref<'a>(
    m: i32,
    n: i32,
    a: *const f32,
    lda: i32,
) -> MatrixRef<'a, f32> { unsafe { 
    assert!(m >= 0 && n >= 0, "m, n must be non-negative");

    if m == 0 || n == 0 {
        let abuf: &'a [f32] = slice::from_raw_parts(a, 0);
        return MatrixRef::new(abuf, 0, 0, 1, 0)
            .expect("MatrixRef::new failed for m = 0 or n = 0");
    }

    assert!(lda > 0, "lda must be positive");
    let m_u   = m as usize;
    let n_u   = n as usize;
    let lda_u = lda as usize;
    let len   = (n_u - 1) * lda_u + m_u;

    let abuf: &'a [f32] = slice::from_raw_parts(a, len);

    MatrixRef::new(abuf, m_u, n_u, lda_u, 0)
        .expect("A view failed")
}}

pub unsafe fn ptr_to_mat_mut<'a>(
    m: i32,
    n: i32,
    a: *mut f32,
    lda: i32,
) -> MatrixMut<'a, f32> { unsafe { 
    assert!(m >= 0 && n >= 0, "m, n must be non-negative");

    if m == 0 || n == 0 {
        let abuf: &'a mut [f32] = slice::from_raw_parts_mut(a, 0);
        return MatrixMut::new(abuf, 0, 0, 1, 0)
            .expect("MatrixMut::new failed for m = 0 or n = 0");
    }

    assert!(lda > 0, "lda must be positive");
    let m_u   = m as usize;
    let n_u   = n as usize;
    let lda_u = lda as usize;
    let len   = (n_u - 1) * lda_u + m_u;

    let abuf: &'a mut [f32] = slice::from_raw_parts_mut(a, len);

    MatrixMut::new(abuf, m_u, n_u, lda_u, 0)
        .expect("A mut view failed")
}}
