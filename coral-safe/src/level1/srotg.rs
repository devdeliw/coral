//! BLAS Level 1 `?ROTG` routine in single precision. 
//!
//! \\[ 
//! \begin{bmatrix} 
//! c & s \\\\ -s & c 
//! \end{bmatrix} \begin{bmatrix} 
//! a \\\\ b 
//! \end{bmatrix} = \begin{bmatrix} 
//! r \\\\ 0 
//! \end{bmatrix}
//! \\]
//!
//! # Author 
//! Deval Deliwala 


/// Computes parameters `c` and `s` that define a Givens 
/// rotation such that the vector [a, b] -> [r, 0] under 
/// the operator [ c s \\ -s c ].
///
/// `z` is an auxilary parameter related to re-constructing the 
/// Givens rotation given `r`. 
///
/// Arguments: 
/// - `a`: `&mut f32` - input scalar, overwritten with `r`
/// - `b`: `&mut f32` - input scalar, overwritten with `z`
/// - `c`: `&mut f32` - output scalar cosine component of Givens rot. 
/// - `s`: `&mut f32` - output scalar sine component of Givens rot.
///
/// Returns: 
/// Nothing. The results overwrite `a`, `b`, `c`, and `s`. 
#[inline] 
pub fn srotg ( 
    a: &mut f32, 
    b: &mut f32, 
    c: &mut f32, 
    s: &mut f32, 
) { 

    let a_abs = a.abs(); 
    let b_abs = b.abs(); 

    let p = if a_abs > b_abs { 
        *a
    } else { 
        *b 
    } ;
    let scale = a_abs + b_abs;  

    // quick return
    if scale == 0.0 { 
        *c = 1.0; 
        *s = 0.0; 
        *a = 0.0; 
        *b = 0.0;
        return;
    }

    let r = { 
        let p1 = *a / scale;
        let p2 = *b / scale;

        p.signum() * scale * p1.hypot(p2) 
    };

    *c = *a / r; 
    *s = *b / r; 
       
    let mut z = 1.0; 
    if a_abs > b_abs { 
        z = *s; 
    } 

    if b_abs >= a_abs && *c != 0.0 { 
        z = 1.0 / *c; 
    }

    *a = r; 
    *b = z;
}
