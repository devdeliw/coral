//! `coral` is a BLAS implementation in pure rust for AArch64.
//! 
//! - no dependencies.
//! - column-major only. 
//! - level1 and level2 routines fully implemented. 
//! - level3 only has `GEMM`. 
//!
//! benchmarks: <https://dev-undergrad.dev/posts/benchmarks/>
//!
//! ## example
//!
//! `SGEMM` 
//! \\[ 
//!     C \ \leftarrow \ \alpha A B + \beta C 
//! \\]
//!
//! ```
//! use coral_aarch64::level3::sgemm;
//! use coral_aarch64::enums::CoralTranspose;
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
//!         1.0, 0.0,
//!         0.0, 1.0,
//!     ];
//!
//!     let m = 2;
//!     let n = 2;
//!     let k = 2;
//!
//!     let alpha = 2.0;
//!     let beta  = 1.0;
//!
//!     sgemm(
//!         CoralTranspose::NoTranspose,
//!         CoralTranspose::NoTranspose,
//!         m, n, k,
//!         alpha,
//!         a.as_ptr(), m,
//!         b.as_ptr(), k,
//!         beta,
//!         c.as_mut_ptr(), m,
//!     );
//!
//!     // C = [[47, 62],
//!     //      [68, 93]]
//!     assert!((c[0] - 47.0).abs() < 1e-6);
//!     assert!((c[1] - 68.0).abs() < 1e-6);
//!     assert!((c[2] - 62.0).abs() < 1e-6);
//!     assert!((c[3] - 93.0).abs() < 1e-6);
//! }
//! ```


#[cfg(target_arch = "aarch64")]
pub mod level1;

#[cfg(target_arch = "aarch64")]
pub mod level2;

#[cfg(target_arch = "aarch64")]
pub mod level3;

pub mod enums;

#[cfg(target_arch = "aarch64")]
pub(crate) mod level1_special;


