pub(crate) mod packers;
pub(crate) mod microkernel;
pub(crate) mod macrokernel;

pub(crate) mod sgemm_nn; 
pub(crate) mod sgemm_nt;
pub(crate) mod sgemm_tn; 
pub(crate) mod sgemm_tt; 

pub(crate) use sgemm_nn::sgemm_nn; 
pub(crate) use sgemm_nt::sgemm_nt; 
pub(crate) use sgemm_tn::sgemm_tn; 

pub mod sgemm;
pub use sgemm::sgemm; 

pub(crate) mod subs_lower;
pub(crate) mod subs_upper; 
pub(crate) mod strlsm_n; 
pub(crate) mod strusm_n;
pub(crate) mod strlsm_t; 
pub(crate) mod strusm_t;

pub mod strsm; 
pub use strsm::strsm; 

