use crate::level1::saxpy; 
use crate::types::{MatrixMut, VectorRef, VectorMut, CoralTriangular}; 


#[inline] 
fn upper ( 
    alpha: f32, 
    lda: usize, 
    x: &[f32], 
    y: &[f32], 
    a: &mut [f32], 
) { 
    let n = x.len();
    for j in 0..n { 
        let aj_y = alpha * y[j]; 
        let aj_x = alpha * x[j]; 
        let col_start = j * lda; 

        if aj_y != 0.0 { 
            let xbuf = &x[0..j + 1]; 
            let ybuf = &mut a[col_start .. col_start + (j + 1)];
            let xview = VectorRef::new(xbuf, j + 1, 1, 0)
                .expect("x make failed"); 
            let yview = VectorMut::new(ybuf, j + 1, 1, 0)
                .expect("y make failed"); 

            saxpy(aj_y, xview, yview); 
        }

        if aj_x != 0.0 { 
            let xbuf = &y[0..j + 1]; 
            let ybuf = &mut a[col_start .. col_start + (j + 1)]; 
            let xview = VectorRef::new(xbuf, j + 1, 1, 0) 
                .expect("x make failed"); 
            let yview = VectorMut::new(ybuf, j + 1, 1, 0)
                .expect("y make failed"); 

            saxpy(aj_x, xview, yview); 
        }
    }
} 

#[inline] 
fn lower ( 
    alpha: f32, 
    lda: usize, 
    x: &[f32], 
    y: &[f32], 
    a: &mut [f32], 
) {
    let n = x.len(); 
    for j in 0..n { 
        let aj_y = alpha * y[j]; 
        let aj_x = alpha * x[j]; 
        let col_start = j * lda + j; 

        if aj_y != 0.0 { 
            let xbuf = &x[j..n]; 
            let ybuf = &mut a[col_start .. j * lda + n]; 
            let xview = VectorRef::new(xbuf, n - j, 1, 0)
                .expect("x make failed"); 
            let yview = VectorMut::new(ybuf, n - j, 1, 0)
                .expect("y make failed"); 

            saxpy(aj_y, xview, yview); 
        }

        if aj_x != 0.0 {
            let xbuf = &y[j..n]; 
            let ybuf = &mut a[col_start .. j * lda + n]; 
            let xview = VectorRef::new(xbuf, n - j, 1, 0)
                .expect("x make failed"); 
            let yview = VectorMut::new(ybuf, n - j, 1, 0)
                .expect("y make failed"); 

            saxpy(aj_x, xview, yview); 
        }
    }
}



#[inline] 
pub fn ssyr2 ( 
    uplo: CoralTriangular, 
    alpha: f32, 
    mut a: MatrixMut<'_, f32>, 
    x: VectorRef<'_, f32>, 
    y: VectorRef<'_, f32>, 
) { 
    debug_assert!(a.compare_m_n(), "n_cols must equal n_rows"); 
    let n = a.n_rows(); 
    debug_assert!(x.compare_n(n), "x must have logical length n"); 
    debug_assert!(y.compare_n(n), "y must have logical length n"); 

    if n == 0 || alpha == 0.0 { 
        return; 
    }

    let incx = x.stride();
    let incy = y.stride(); 
    let xoff = x.offset(); 
    let yoff = y.offset(); 
    let aoff = a.offset(); 
    let lda = a.lda(); 

    let xs = x.as_slice(); 
    let ys = y.as_slice(); 
    let aslice = a.as_slice_mut(); 

    let xs = &xs[xoff..]; 
    let ys = &ys[yoff..]; 
    let aslice = &mut aslice[aoff..];

    // fast path 
    if incx == 1 && incy == 1 { 
        match uplo { 
           CoralTriangular::Upper => upper(alpha, lda, xs, ys, aslice), 
           CoralTriangular::Lower => lower(alpha, lda, xs, ys, aslice), 
        }

        return; 
    }

    // slow path 
    match uplo {
        CoralTriangular::Upper => {
            for j in 0..n {
                let aj_y = alpha * ys[j * incy];
                let aj_x = alpha * xs[j * incx];

                if aj_y == 0.0 && aj_x == 0.0 {
                    continue;
                }

                let col_start = j * lda;
                let mut a_idx = col_start;

                let mut i = 0;
                while i <= j {
                    let xi = xs[i * incx];
                    let yi = ys[i * incy];

                    let a_ij = &mut aslice[a_idx];
                    *a_ij = xi.mul_add(aj_y, *a_ij);
                    *a_ij = yi.mul_add(aj_x, *a_ij);

                    a_idx += 1; 
                    i += 1;
                }
            }
        }
        CoralTriangular::Lower => {
            for j in 0..n {
                let aj_y = alpha * ys[j * incy];
                let aj_x = alpha * xs[j * incx];

                if aj_y == 0.0 && aj_x == 0.0 {
                    continue;
                }

                let mut a_idx = j * lda + j;

                let mut i = j;
                while i < n {
                    let xi = xs[i * incx];
                    let yi = ys[i * incy];

                    let a_ij = &mut aslice[a_idx];
                    *a_ij = xi.mul_add(aj_y, *a_ij);
                    *a_ij = yi.mul_add(aj_x, *a_ij);

                    a_idx += 1;
                    i += 1;
                }
            }
        }
    }
}
