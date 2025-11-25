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

Core Rust Architecture for Linear Algebra. This is a pure-Rust implementation of BLAS routines 
that is *fully* `safe`, and uses *portable-simd*; it's applicable for all architectures and has no
dependencies. 

`coral` uses a more idiomatic, modern API to be fully-safe. However, an `unsafe` fortran77 wrapper conforming 
to the legacy BLAS API is also provided. 

Only single-precision routines are implemented, and it needs nightly. 
