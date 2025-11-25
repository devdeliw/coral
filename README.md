<p align="center">
  <img src="./coral_logo.png" width="75%">
</p>

<p align="center">
  <a href="https://crates.io/crates/coral-aarch64">
    <img src="https://img.shields.io/crates/v/coral-aarch64.svg?style=flat-square" alt="crates.io">
  </a>
  <a href="https://docs.rs/coral-aarch64">
    <img src="https://docs.rs/coral-aarch64/badge.svg?style=flat-square" alt="docs.rs">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="license">
  </a>
</p>

## CORAL 

Core Rust Architecture for Linear Algebra. This is a pure-Rust implementation of BLAS routines 
that is optimized and uses NEON AArch64 intrinsics for speed and has no dependencies. However, it is scarily `unsafe`, as it works 
with pointers and `unsafe` simd. 

While it is faster and more comparable with `OpenBLAS` and [faer](https://faer.veganb.tw), it is not 
that far off from the *fully*-safe and *portable-simd* [coral](https://docs.rs/coral-blas/latest/coral/) 
implementation. For these reasons, unless you currently need double precision or complex routines, or 
need an extra 10GFLOP/s on `SGEMM`, I highly suggest using the safe implementation.
