use crate::level2::strmv; 
use crate::types::{CoralDiagonal, CoralTranspose, CoralTriangular}; 
use crate::fortran::wrappers::{ptr_to_view_mut, ptr_to_mat_ref}; 

/// unsafe wrapper for [strmv] routine
#[inline] 
pub unsafe fn strmv_f77 ( 
    uplo: CoralTriangular, 
    op: CoralTranspose, 
    diag: CoralDiagonal, 
    n: i32, 
    a: *const f32, 
    lda: i32, 
    x: *mut f32, 
    incx: i32,
) { unsafe { 
    let xview = ptr_to_view_mut(n, x, incx); 
    let aview = ptr_to_mat_ref(n, n, a, lda); 

    strmv(uplo, op, diag, aview, xview); 
}}
