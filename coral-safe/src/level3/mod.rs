pub(crate) mod packers;
pub(crate) mod microkernel;
pub(crate) mod macrokernel;

pub(crate) mod sgemm_nn; 
pub(crate) mod sgemm_nt;
pub(crate) mod sgemm_tn; 
pub(crate) mod sgemm_tt; 
pub mod sgemm;
pub use sgemm::sgemm; 

pub(crate) mod substitutions;
pub(crate) mod strlsm_n; 
pub(crate) mod strusm_n;
pub(crate) mod strlsm_t; 
pub(crate) mod strusm_t;

