<p align="center">
  <img src="./coral_logo.png" width="75%">
</p>

<p align="center">
  <a href="https://crates.io/crates/coral-blas">
    <img src="https://img.shields.io/crates/v/coral-blas.svg?style=flat-square" alt="crates.io">
  </a>
  <a href="https://docs.rs/coral-blas">
    <img src="https://docs.rs/coral-blas/badge.svg?style=flat-square" alt="docs.rs">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="license">
  </a>
</p>

## CORAL 

Core Rust Architecture for Linear Algebra. This is a pure-Rust implementation of BLAS routines. 

There are two versions: `coral` and `coral-aarch64`. 

The first is a nightly, *fully* `safe`, and 
`portable-simd` version with a more idiomatic API. It has an an `unsafe` fortran77 wrapper if needed too.
The second is a very `unsafe`, but *slightly* faster BLAS for AArch64 only. In most cases, `coral` should be used.

Here are some [benchmarks](https://dev-undergrad.dev/posts/benchmarks/). 
