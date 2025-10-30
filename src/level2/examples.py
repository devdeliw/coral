examples = {
  "sgemv.rs": """//! 
//! # Example
//! use coral::level2::sgemv::sgemv;
//!
//! fn main() {
//!     // y := alpha A x + beta y  (m=2, n=3), column-major A
//!     let m = 2;
//!     let n = 3;
//!     let a = vec![
//!         1.0, 2.0,  // col 0
//!         3.0, 4.0,  // col 1
//!         5.0, 6.0,  // col 2
//!     ];
//!
//!     let lda   = m;
//!     let x     = vec![1.0, 2.0, 3.0];
//!     let incx  = 1;
//!     let mut y = vec![0.5, -1.0];
//!     let incy  = 1;
//!     let alpha = 2.0;
//!     let beta  = 0.5;
//!
//!     sgemv(m, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
//! }""",

  "dgemv.rs": """//! 
//! # Example
//! use coral::level2::dgemv::dgemv;
//!
//! fn main() {
//!     let m = 2;
//!     let n = 2;
//!     let a = vec![
//!         1.0, 4.0,  // col 0
//!         2.0, 5.0,  // col 1
//!     ];
//!
//!     let lda   = m;
//!     let x     = vec![1.0, -1.0];
//!     let incx  = 1;
//!     let mut y = vec![0.0, 0.0];
//!     let incy  = 1;
//!     let alpha = 1.0;
//!     let beta  = 0.0;
//!
//!     dgemv(m, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
//! }""",

  "cgemv.rs": """//! 
//! # Example
//! use coral::level2::cgemv::cgemv;
//!
//! fn main() {
//!     // interleaved complex, m=2, n=2
//!     let m = 2;
//!     let n = 2;
//!     let a = vec![
//!         1.0, 0.0,  0.0, 1.0,  // col 0: (1+0i, 0+1i)
//!         2.0, 0.0,  0.0, 0.0,  // col 1: (2+0i, 0+0i)
//!     ];
//!
//!     let lda   = m;
//!     let x     = vec![1.0, 1.0,  0.0, -1.0];  // (1+i, -i)
//!     let incx  = 1;
//!     let mut y = vec![0.0, 0.0,  0.0, 0.0];   // 2 outputs
//!     let incy  = 1;
//!     let alpha = [1.0, 0.0];
//!     let beta  = [0.0, 0.0];
//!
//!     cgemv(m, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
//! }""",

  "zgemv.rs": """//! 
//! # Example
//! use coral::level2::zgemv::zgemv;
//!
//! fn main() {
//!     let m = 2;
//!     let n = 2;
//!     let a = vec![
//!         1.0, 0.0,  0.0, 1.0,
//!         2.0, 0.0,  0.0, 0.0,
//!     ];
//!
//!     let lda   = m;
//!     let x     = vec![1.0, -1.0,  0.5, 0.5]; // (1 - i, 0.5 + 0.5i)
//!     let incx  = 1;
//!     let mut y = vec![0.0, 0.0,  0.0, 0.0];
//!     let incy  = 1;
//!     let alpha = [1.0, 0.0];
//!     let beta  = [0.0, 0.0];
//!
//!     zgemv(m, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
//! }""",

  "strsv.rs": """//! 
//! # Example
//! use coral::level2::strsv::strsv;
//! use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
//!
//! fn main() {
//!     let n         = 2;
//!     let uplo      = CoralTriangular::UpperTriangular;
//!     let transpose = CoralTranspose::NoTranspose;
//!     let diagonal  = CoralDiagonal::NonUnitDiagonal;
//!     let a = vec![
//!         2.0, 0.0,  // col 0
//!         1.0, 3.0,  // col 1
//!     ];
//!
//!     let lda  = n;
//!     let mut x = vec![2.0, 3.0]; // b on entry; overwritten with solution
//!     let incx = 1;
//!
//!     strsv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
//! }""",

  "dtrsv.rs": """//! 
//! # Example
//! use coral::level2::dtrsv::dtrsv;
//! use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
//!
//! fn main() {
//!     let n         = 2;
//!     let uplo      = CoralTriangular::UpperTriangular;
//!     let transpose = CoralTranspose::NoTranspose;
//!     let diagonal  = CoralDiagonal::NonUnitDiagonal;
//!     let a = vec![
//!         2.0, 0.0,
//!         1.0, 3.0,
//!     ];
//!
//!     let lda  = n;
//!     let mut x = vec![2.0, 3.0];
//!     let incx = 1;
//!
//!     dtrsv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
//! }""",

  "ctrsv.rs": """//! 
//! # Example
//! use coral::level2::ctrsv::ctrsv;
//! use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
//!
//! fn main() {
//!     // interleaved complex triangular solve
//!     let n         = 2;
//!     let uplo      = CoralTriangular::UpperTriangular;
//!     let transpose = CoralTranspose::NoTranspose;
//!     let diagonal  = CoralDiagonal::NonUnitDiagonal;
//!     let a = vec![
//!         2.0, 0.0,   0.0, 0.0,  // col 0: (2+0i, 0+0i)
//!         1.0, 1.0,   3.0, 0.0,  // col 1: (1+i, 3+0i)
//!     ];
//!
//!     let lda   = n;
//!     let mut x = vec![3.0, 0.0,  1.0, 0.0]; // b -> solution (interleaved)
//!     let incx  = 1;
//!
//!     ctrsv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
//! }""",

  "ztrsv.rs": """//! 
//! # Example
//! use coral::level2::ztrsv::ztrsv;
//! use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
//!
//! fn main() {
//!     let n         = 2;
//!     let uplo      = CoralTriangular::UpperTriangular;
//!     let transpose = CoralTranspose::NoTranspose;
//!     let diagonal  = CoralDiagonal::NonUnitDiagonal;
//!     let a = vec![
//!         2.0, 0.0,   0.0, 0.0,
//!         1.0, -1.0,  3.0, 0.0,
//!     ];
//!
//!     let lda   = n;
//!     let mut x = vec#[ 2.0, 1.0,  1.0, 0.0 ]; // b -> solution
//!     let incx  = 1;
//!
//!     ztrsv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
//! }""".replace("#[", "["),  # avoid confusing the doc generator

  "strmv.rs": """//! 
//! # Example
//! use coral::level2::strmv::strmv;
//! use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
//!
//! fn main() {
//!     let n         = 2;
//!     let uplo      = CoralTriangular::UpperTriangular;
//!     let transpose = CoralTranspose::NoTranspose;
//!     let diagonal  = CoralDiagonal::NonUnitDiagonal;
//!     let a = vec![
//!         2.0, 0.0,
//!         1.0, 3.0,
//!     ];
//!
//!     let lda  = n;
//!     let mut x = vec![1.0, 2.0];
//!     let incx = 1;
//!
//!     strmv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
//! }""",

  "dtrmv.rs": """//! 
//! # Example
//! use coral::level2::dtrmv::dtrmv;
//! use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
//!
//! fn main() {
//!     let n         = 2;
//!     let uplo      = CoralTriangular::LowerTriangular;
//!     let transpose = CoralTranspose::NoTranspose;
//!     let diagonal  = CoralDiagonal::NonUnitDiagonal;
//!     let a = vec![
//!         2.0, 1.0,
//!         0.0, 3.0,
//!     ];
//!
//!     let lda  = n;
//!     let mut x = vec![1.0, 2.0];
//!     let incx = 1;
//!
//!     dtrmv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
//! }""",

  "ctrmv.rs": """//! 
//! # Example
//! use coral::level2::ctrmv::ctrmv;
//! use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
//!
//! fn main() {
//!     let n         = 2;
//!     let uplo      = CoralTriangular::UpperTriangular;
//!     let transpose = CoralTranspose::NoTranspose;
//!     let diagonal  = CoralDiagonal::NonUnitDiagonal;
//!     let a = vec![
//!         2.0, 0.0,   0.0, 0.0,
//!         1.0, 1.0,   3.0, 0.0,
//!     ];
//!
//!     let lda   = n;
//!     let mut x = vec![1.0, 0.0,  2.0, 0.0];
//!     let incx  = 1;
//!
//!     ctrmv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
//! }""",

  "ztrmv.rs": """//! 
//! # Example
//! use coral::level2::ztrmv::ztrmv;
//! use coral::enums::{CoralTriangular, CoralTranspose, CoralDiagonal};
//!
//! fn main() {
//!     let n         = 2;
//!     let uplo      = CoralTriangular::UpperTriangular;
//!     let transpose = CoralTranspose::NoTranspose;
//!     let diagonal  = CoralDiagonal::NonUnitDiagonal;
//!     let a = vec![
//!         2.0, 0.0,   0.0, 0.0,
//!         1.0, -1.0,  3.0, 0.0,
//!     ];
//!
//!     let lda   = n;
//!     let mut x = vec![1.0, 0.0,  2.0, 0.0];
//!     let incx  = 1;
//!
//!     ztrmv(uplo, transpose, diagonal, n, &a, lda, &mut x, incx);
//! }""",

  "ssyr.rs": """//! 
//! # Example
//! use coral::level2::ssyr::ssyr;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     let uplo = CoralTriangular::UpperTriangular;
//!     let n    = 3;
//!     let alpha = 2.0;
//!     let x    = vec![1.0, 2.0, 3.0];
//!     let incx = 1;
//!     let mut a = vec![0.0; n * n]; // column-major; only 'uplo' triangle is referenced
//!     let lda  = n;
//!
//!     ssyr(uplo, n, alpha, &x, incx, &mut a, lda);
//! }""",

  "dsyr.rs": """//! 
//! # Example
//! use coral::level2::dsyr::dsyr;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     let uplo  = CoralTriangular::LowerTriangular;
//!     let n     = 2;
//!     let alpha = 1.0;
//!     let x     = vec#[ 1.0, 2.0 ];
//!     let incx  = 1;
//!     let mut a = vec![0.0; n * n];
//!     let lda   = n;
//!
//!     dsyr(uplo, n, alpha, &x, incx, &mut a, lda);
//! }""".replace("#[", "["),

  "cher.rs": """//! 
//! # Example
//! use coral::level2::cher::cher;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     // Hermitian rank-1: A := alpha x x^H + A  (alpha real)
//!     let uplo  = CoralTriangular::UpperTriangular;
//!     let n     = 2;
//!     let alpha = 0.5;                 // real
//!     let x     = vec![1.0, 0.0,  0.0, 1.0]; // (1+0i, 0+1i)
//!     let incx  = 1;
//!     let mut a = vec![0.0; 2 * n * n]; // interleaved complex storage
//!
//!     let lda = n;
//!     cher(uplo, n, alpha, &x, incx, &mut a, lda);
//! }""",

  "zher.rs": """//! 
//! # Example
//! use coral::level2::zher::zher;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     let uplo  = CoralTriangular::LowerTriangular;
//!     let n     = 2;
//!     let alpha = 1.0;                       // real
//!     let x     = vec![1.0, -1.0,  0.0, 1.0]; // (1 - i, i)
//!     let incx  = 1;
//!     let mut a = vec![0.0; 2 * n * n];
//!
//!     let lda = n;
//!     zher(uplo, n, alpha, &x, incx, &mut a, lda);
//! }""",

  "ssyr2.rs": """//! 
//! # Example
//! use coral::level2::ssyr2::ssyr2;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     let uplo  = CoralTriangular::UpperTriangular;
//!     let n     = 3;
//!     let alpha = 0.5;
//!     let x     = vec![1.0, 2.0, 3.0];
//!     let incx  = 1;
//!     let y     = vec![2.0, -1.0, 0.5];
//!     let incy  = 1;
//!     let mut a = vec![0.0; n * n];
//!     let lda   = n;
//!
//!     ssyr2(uplo, n, alpha, &x, incx, &y, incy, &mut a, lda);
//! }""",

  "dsyr2.rs": """//! 
//! # Example
//! use coral::level2::dsyr2::dsyr2;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     let uplo  = CoralTriangular::LowerTriangular;
//!     let n     = 2;
//!     let alpha = 1.0;
//!     let x     = vec![1.0, 2.0];
//!     let incx  = 1;
//!     let y     = vec#[ 3.0, 4.0 ];
//!     let incy  = 1;
//!     let mut a = vec![0.0; n * n];
//!     let lda   = n;
//!
//!     dsyr2(uplo, n, alpha, &x, incx, &y, incy, &mut a, lda);
//! }""".replace("#[", "["),

  "cher2.rs": """//! 
//! # Example
//! use coral::level2::cher2::cher2;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     // Hermitian rank-2: A := alpha x y^H + conj(alpha) y x^H + A
//!     let uplo  = CoralTriangular::UpperTriangular;
//!     let n     = 2;
//!     let alpha = [0.5, -0.25];                 // complex alpha (re, im)
//!     let x     = vec![1.0, 0.0,   0.0, 1.0];   // (1+0i, 0+1i)
//!     let incx  = 1;
//!     let y     = vec![0.0, 1.0,   1.0, 0.0];   // (i, 1)
//!     let incy  = 1;
//!     let mut a = vec![0.0; 2 * n * n];
//!     let lda   = n;
//!
//!     cher2(uplo, n, alpha, &x, incx, &y, incy, &mut a, lda);
//! }""",

  "zher2.rs": """//! 
//! # Example
//! use coral::level2::zher2::zher2;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     let uplo  = CoralTriangular::LowerTriangular;
//!     let n     = 2;
//!     let alpha = [1.0, 0.5];                  // complex alpha (re, im)
//!     let x     = vec![1.0, -1.0,  0.0, 1.0];  // (1 - i, i)
//!     let incx  = 1;
//!     let y     = vec![0.5, 0.0,  1.0, 0.0];   // (0.5, 1)
//!     let incy  = 1;
//!     let mut a = vec![0.0; 2 * n * n];
//!     let lda   = n;
//!
//!     zher2(uplo, n, alpha, &x, incx, &y, incy, &mut a, lda);
//! }""",

  "cgerc.rs": """//! 
//! # Example
//! use coral::level2::cgerc::cgerc;
//!
//! fn main() {
//!     // A := alpha x y^H + A
//!     let m = 2;
//!     let n = 2;
//!     let alpha = [0.5, 0.0];
//!     let x = vec![1.0, 0.0,  0.0, 1.0];  // (1, i)
//!     let incx = 1;
//!     let y = vec![1.0, -1.0,  2.0, 0.0]; // (1 - i, 2)
//!     let incy = 1;
//!     let mut a = vec![0.0; 2 * m * n];
//!     let lda = m;
//!
//!     cgerc(m, n, alpha, &x, incx, &y, incy, &mut a, lda);
//! }""",

  "cgeru.rs": """//! 
//! # Example
//! use coral::level2::cgeru::cgeru;
//!
//! fn main() {
//!     // A := alpha x y^T + A  (no conjugation on y)
//!     let m = 2;
//!     let n = 2;
//!     let alpha = [0.5, 0.0];
//!     let x = vec![1.0, 0.0,  0.0, 1.0];
//!     let incx = 1;
//!     let y = vec![1.0, -1.0,  2.0, 0.0];
//!     let incy = 1;
//!     let mut a = vec#[ 0.0; 2 * m * n ];
//!     let lda = m;
//!
//!     cgeru(m, n, alpha, &x, incx, &y, incy, &mut a, lda);
//! }""".replace("#[", "["),

  "zgerc.rs": """//! 
//! # Example
//! use coral::level2::zgerc::zgerc;
//!
//! fn main() {
//!     let m = 2;
//!     let n = 2;
//!     let alpha = [1.0, 0.0];
//!     let x = vec![1.0, -1.0,  0.0, 1.0]; // (1 - i, i)
//!     let incx = 1;
//!     let y = vec#[ 0.5, 0.5,  1.0, 0.0 ]; // (0.5 + 0.5i, 1)
//!     let incy = 1;
//!     let mut a = vec![0.0; 2 * m * n];
//!     let lda = m;
//!
//!     zgerc(m, n, alpha, &x, incx, &y, incy, &mut a, lda);
//! }""".replace("#[", "["),

  "zgeru.rs": """//! 
//! # Example
//! use coral::level2::zgeru::zgeru;
//!
//! fn main() {
//!     let m = 2;
//!     let n = 2;
//!     let alpha = [1.0, -0.5];
//!     let x = vec![1.0, 0.0,  2.0, 0.0];  // (1, 2)
//!     let incx = 1;
//!     let y = vec![0.0, 1.0,  1.0, 0.0];  // (i, 1)
//!     let incy = 1;
//!     let mut a = vec![0.0; 2 * m * n];
//!     let lda = m;
//!
//!     zgeru(m, n, alpha, &x, incx, &y, incy, &mut a, lda);
//! }""",

  "chemv.rs": """//! 
//! # Example
//! use coral::level2::chemv::chemv;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     // y := alpha A x + beta y, A Hermitian
//!     let uplo  = CoralTriangular::UpperTriangular;
//!     let n     = 2;
//!     let alpha = [1.0, 0.0];
//!     let a = vec![
//!         2.0, 0.0,   0.0, -1.0,  // col 0: (2, -i)
//!         0.0, 1.0,   3.0,  0.0,  // col 1: (i, 3)
//!     ];
//!
//!     let lda   = n;
//!     let x     = vec![1.0, 0.0,  0.0, 1.0];   // (1, i)
//!     let incx  = 1;
//!     let beta  = [0.0, 0.0];
//!     let mut y = vec![0.0, 0.0,  0.0, 0.0];
//!     let incy  = 1;
//!
//!     chemv(uplo, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
//! }""",

  "zhemv.rs": """//! 
//! # Example
//! use coral::level2::zhemv::zhemv;
//! use coral::enums::CoralTriangular;
//!
//! fn main() {
//!     let uplo  = CoralTriangular::LowerTriangular;
//!     let n     = 2;
//!     let alpha = [1.0, 0.0];
//!     let a = vec![
//!         2.0, 0.0,   0.0,  1.0,
//!         0.0, -1.0,  3.0,  0.0,
//!     ];
//!
//!     let lda   = n;
//!     let x     = vec![1.0, -1.0,  0.0, 1.0];
//!     let incx  = 1;
//!     let beta  = [0.0, 0.0];
//!     let mut y = vec#[ 0.0, 0.0,  0.0, 0.0 ];
//!     let incy  = 1;
//!
//!     zhemv(uplo, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
//! }""".replace("#[", "["),
}

