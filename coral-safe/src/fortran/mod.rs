pub(crate) mod helpers;
pub mod index;

// level 1
pub mod isamax;
pub mod sasum;
pub mod saxpy;
pub mod scopy;
pub mod sdot;
pub mod snrm2;
pub mod srot;
pub mod srotg;
pub mod srotm;
pub mod srotmg;
pub mod sswap;

pub use isamax::{isamax_lp64, isamax_ilp64};
pub use sasum::{sasum_lp64, sasum_ilp64};
pub use saxpy::{saxpy_lp64, saxpy_ilp64};
pub use scopy::{scopy_lp64, scopy_ilp64};
pub use sdot::{sdot_lp64, sdot_ilp64};
pub use snrm2::{snrm2_lp64, snrm2_ilp64};
pub use srot::{srot_lp64, srot_ilp64};
pub use srotg::srotg_f77;
pub use srotm::{srotm_lp64, srotm_ilp64};
pub use srotmg::srotmg_f77;
pub use sswap::{sswap_lp64, sswap_ilp64};

// level 2
pub mod sgemv;
pub mod sger;
pub mod ssymv;
pub mod ssyr;
pub mod ssyr2;
pub mod strmv;
pub mod strsv;

pub use sgemv::{sgemv_lp64, sgemv_ilp64};
pub use sger::{sger_lp64, sger_ilp64};
pub use ssymv::{ssymv_lp64, ssymv_ilp64};
pub use ssyr::{ssyr_lp64, ssyr_ilp64};
pub use ssyr2::{ssyr2_lp64, ssyr2_ilp64};
pub use strmv::{strmv_lp64, strmv_ilp64};
pub use strsv::{strsv_lp64, strsv_ilp64};

// level 3
pub mod sgemm;
pub use sgemm::{sgemm_lp64, sgemm_ilp64};

