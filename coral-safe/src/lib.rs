//! `coral` is a fully-safe, portable BLAS implementation in pure rust.
//! 
//! - no dependencies.
//! - is nightly.
//! - column-major only. 
//! - single precision only. 
//! - level3 only has `sgemm`. 
//!
//! benchmarks: <https://dev-undergrad.dev/posts/benchmarks/>
//!
//! ## example
//!
//! `sgemm` 
//! \\[ 
//!     C \ \leftarrow \ \alpha A B + \beta C 
//! \\]
//!
//! ```
//! use coral::level3::sgemm;
//! use coral::types::{MatrixRef, MatrixMut, CoralTranspose};
//!
//! fn main() {
//!     // A = [[1, 3],
//!     //      [2, 4]]
//!     let a = vec![
//!         1.0, 2.0,   // column 0
//!         3.0, 4.0,   // column 1
//!     ];
//!
//!     // B = [[5, 7],
//!     //      [6, 8]]
//!     let b = vec![
//!         5.0, 6.0,   // column 0
//!         7.0, 8.0,   // column 1
//!     ];
//!
//!     // C = identity
//!     let mut c = vec![
//!         1.0, 0.0,   // column 0 
//!         0.0, 1.0,   // column 1
//!     ];
//!
//!     let m = 2;
//!     let n = 2;
//!     let k = 2;
//!
//!     let lda = m; 
//!     let ldb = n; 
//!     let ldc = m; 
//!
//!     let aview = MatrixRef::new(&a, m, k, lda, 0)
//!         .expect("matrix ref build failed"); 
//!     let bview = MatrixRef::new(&b, k, n, ldb, 0)
//!         .expect("matrix ref build failed"); 
//!     let cview = MatrixMut::new(&mut c, m, n, ldc, 0)
//!         .expect("matrix mut build failed");
//!
//!     let alpha = 2.0;
//!     let beta  = 1.0;
//!
//!     sgemm( 
//!         CoralTranspose::NoTrans, 
//!         CoralTranspose::NoTrans, 
//!         alpha, 
//!         beta, 
//!         aview, 
//!         bview, 
//!         cview,
//!     );
//!
//!     let cdata = c.as_slice(); 
//!
//!     // C = [[47, 62],
//!     //      [68, 93]]
//!     assert!((cdata[0] - 47.0).abs() < 1e-6);
//!     assert!((cdata[1] - 68.0).abs() < 1e-6);
//!     assert!((cdata[2] - 62.0).abs() < 1e-6);
//!     assert!((cdata[3] - 93.0).abs() < 1e-6);
//! }
//! ```

#![feature(portable_simd)]
pub mod level1;
pub mod fused; 
pub mod level2; 
pub mod level3;
pub mod fortran; 

pub mod errors;
pub mod types;



