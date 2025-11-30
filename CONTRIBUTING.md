# Contributing

This describes how to contribute to `coral-safe` to help build a safe rust BLAS. 

## What to work on

Current useful areas:

- Implementing double-precision and complex-precision routines analogous to the existing single-precision ones.
- Finishing the remaining Level 3 single-precision routines.

## Workflow

1. Fork the repo on GitHub.
3. Make your changes or add new routines.
4. Add or update tests for any new behavior by comparing to `cblas`.
5. Run the tests from `coral-safe/`:

   ```bash
   cargo test --features openblas 
   ``` 

6. Open a pull request against the main branch with a short description of the change.

## Constraints 

- Keep naming and APIs consistent with existing single-precision routines. Understand the types `coral-safe` 
already has. 
- It should remain fully-safe Rust; aoid using `unsafe` code here. 
- For new BLAS routines, cover a range of sizes and edge cases 
(e.g. `n = 0`, non-unit strides, non-square matrices, etc) in tests. 

