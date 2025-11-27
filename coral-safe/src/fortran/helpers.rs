use std::slice; 
use crate::types::{VectorRef, VectorMut, MatrixRef, MatrixMut}; 
use crate::fortran::index::BlasIdx; 

pub unsafe fn ptr_to_vec_ref<'a, I: BlasIdx> ( 
    n: I, 
    x: *const f32, 
    incx: I, 
) -> VectorRef<'a, f32> { unsafe { 
    let n_u = n.to_usize(); 
    let incx_u = incx.to_usize(); 

    if n_u == 0 { 
        let xbuf: &'a [f32] = slice::from_raw_parts(x, 0); 

        // with 0 logical elements, defaults to incx = 1 
        return VectorRef::new(xbuf, 0, 1, 0)
            .expect("VectorRef::new failed for n=0"); 
    }

    // required length for given incx 
    let len = 1 + (n_u - 1) * incx_u; 
    let xbuf: &'a [f32] = slice::from_raw_parts(x, len); 

    // internally validates 
    VectorRef::new(xbuf, n_u, incx_u, 0)
        .expect("VectorRef::new failed")
}}

pub unsafe fn ptr_to_vec_mut<'a, I: BlasIdx> ( 
    n: I, 
    x: *mut f32, 
    incx: I, 
) -> VectorMut<'a, f32> { unsafe { 
    let n_u = n.to_usize(); 
    let incx_u = incx.to_usize(); 

    if n_u == 0 { 
        let xbuf: &'a mut [f32] = slice::from_raw_parts_mut(x, 0);

        // with 0 logical elements, defaults to incx = 1 
        return VectorMut::new(xbuf, 0, 1, 0)
            .expect("VectorMut::new failed for n=0"); 
    }

    // required length for given incx 
    let len = 1 + (n_u - 1) * incx_u; 
    let xbuf: &'a mut [f32] = slice::from_raw_parts_mut(x, len); 

    VectorMut::new(xbuf, n_u, incx_u, 0)
        .expect("VectorMut::new failed") 
}}

pub unsafe fn ptr_to_mat_ref<'a, I: BlasIdx> ( 
    m: I, 
    n: I, 
    a: *const f32, 
    lda: I, 
) -> MatrixRef<'a, f32> { unsafe { 
    let m_u = m.to_usize(); 
    let n_u = n.to_usize(); 
    let lda_u = lda.to_usize(); 

    if m_u == 0 || n_u == 0 { 
        let abuf: &'a [f32] = slice::from_raw_parts(a, 0); 

        // with 0 logical elements, defaults to incx = 1 
        return MatrixRef::new(abuf, 0, 0, 1, 0)
            .expect("MatrixRef::new failed for m = 0 or n = 0"); 
    }

    // required length for given lda, n, m 
    let len = (n_u - 1) * lda_u + m_u; 
    let abuf: &'a [f32] = slice::from_raw_parts(a, len); 

    MatrixRef::new(abuf, m_u, n_u, lda_u, 0) 
        .expect("MatrixRef::new failed") 
}}

pub unsafe fn ptr_to_mat_mut<'a, I: BlasIdx> ( 
    m: I, 
    n: I, 
    a: *mut f32, 
    lda: I, 
) -> MatrixMut<'a, f32> { unsafe { 
    let m_u = m.to_usize(); 
    let n_u = n.to_usize(); 
    let lda_u = lda.to_usize(); 

    if m_u == 0 || n_u == 0 { 
        let abuf: &'a mut [f32] = slice::from_raw_parts_mut(a, 0); 

        // with 0 logical elements, defaults to incx = 1 
        return MatrixMut::new(abuf, 0, 0, 1, 0)
            .expect("MatrixMut::new failed for m = 0 or n = 0"); 
    }

    // required length for given lda, n, m 
    let len = (n_u - 1) * lda_u + m_u; 
    let abuf: &'a mut [f32] = slice::from_raw_parts_mut(a, len); 

    MatrixMut::new(abuf, m_u, n_u, lda_u, 0) 
        .expect("MatrixMut::new failed") 
}}



