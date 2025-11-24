//! Level 2 [`?SYR`](https://netlib.org/lapack//explore-html/dc/d82/group__her_gad7585662770cdd3001ed08c7a864cd21.html)
//! routine in single precision. 
//!
//! \\[ 
//! A \leftarrow \alpha x x^T + A. 
//! \\]
//!
//! # Author 
//! Deval Deliwala 


use crate::level1::saxpy; 
use crate::types::CoralTriangular; 
use crate::types::{MatrixMut, VectorRef, VectorMut}; 


#[inline] 
fn upper ( 
    alpha: f32, 
    lda: usize, 
    a: &mut [f32], 
    x: &[f32],
) { 
    for j in 0..x.len() { 
        let aj = alpha * x[j]; 

        if aj != 0.0 { 
            let col_start = j * lda; 

            let xbuf = &x[..j + 1];
            let ybuf = &mut a[col_start .. col_start + (j + 1)]; 
            let xview = VectorRef::new(xbuf, j + 1, 1, 0)
                .expect("x make failed"); 
            let yview = VectorMut::new(ybuf, j + 1, 1, 0)
                .expect("y make failed"); 
            
            saxpy(aj, xview, yview);
        }
    }
}

#[inline] 
fn lower ( 
    alpha: f32, 
    lda: usize, 
    a: &mut [f32], 
    x: &[f32], 
) { 
    let n = x.len(); 
    for j in 0..x.len() { 
        let aj = alpha * x[j]; 
        
        if aj != 0.0 { 
            let col_start = j * lda + j; 


            let xbuf = &x[j..]; 
            let ybuf = &mut a[col_start .. j * lda + n]; 
            let xview = VectorRef::new(xbuf, n - j, 1, 0)
                .expect("x make failed"); 
            let yview = VectorMut::new(ybuf, n - j, 1, 0)
                .expect("y make failed"); 

            saxpy(aj, xview, yview); 
        }
    }
}


/// Performs a symmetric rank-1 update. 
/// ` A += alpha x x^T`
///
/// Arguments: 
/// * `uplo`: [CoralTriangular] - which triangle to reference 
/// * `alpha`: [f32] - scalar for `alpha x x^T` 
/// * `a`: [MatrixMut] - over [f32], symmetric `n x n` 
/// * `x`: [VectorMut] - over [f32]
///
/// Returns: 
/// Nothing. `a.data` is updated in place. 
#[inline] 
pub fn ssyr ( 
    uplo: CoralTriangular, 
    alpha: f32, 
    mut a: MatrixMut<'_, f32>, 
    x: VectorRef<'_, f32>, 
) { 
    debug_assert!(a.compare_m_n(), "n_rows must equal n_cols"); 
    let n = a.n_rows(); 
    debug_assert!(x.compare_n(n), "x must have n logical elements"); 
    
    if n == 0 || alpha == 0.0 { 
        return; 
    }

    let incx  = x.stride();
    let xoff  = x.offset();
    let lda   = a.lda();
    let xs    = x.as_slice();
    let aoff  = a.offset(); 
    let aslice = a.as_slice_mut();
    let aslice = &mut aslice[aoff..];

    // fast path
    if incx == 1 { 
        match uplo { 
            CoralTriangular::Upper => upper(alpha, lda, aslice, &xs[xoff..]), 
            CoralTriangular::Lower => lower(alpha, lda, aslice, &xs[xoff..]), 
        }

        return; 
    }

    // slow path 
    match uplo {
        CoralTriangular::Upper => {
            for j in 0..n {
                let xj = xs[xoff + j * incx];
                let aj = alpha * xj;

                if aj != 0.0 {
                    let col_base = j * lda;

                    let mut xi_idx = xoff;

                    for row in 0..=j {
                        let a_idx = col_base + row;
                        let xi    = xs[xi_idx];

                        // A[i, j] = A[i,j] + aj * x[i]
                        aslice[aoff + a_idx] = xi.mul_add(aj, aslice[aoff + a_idx]);
                        xi_idx += incx;
                    }
                }
            }
        }, 
        CoralTriangular::Lower => {
            for j in 0..n {
                let xj = xs[xoff + j * incx];
                let aj = alpha * xj;

                if aj != 0.0 {
                    let col_base = j * lda;

                    let mut xi_idx = xoff + j * incx;

                    for row in j..n {
                        let a_idx = col_base + row;
                        let xi    = xs[xi_idx];

                        aslice[a_idx] = xi.mul_add(aj, aslice[a_idx]);
                        xi_idx += incx;
                    }
                }
            }
        }
    }
}

