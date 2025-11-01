#[cfg(not(target_arch = "aarch64"))]
compile_error!("coral-blas targets AArch64 only. \
This build is running on a non-AArch64 target.");

#[cfg(target_arch = "aarch64")]
#[cfg_attr(docsrs, doc(cfg(target_arch = "aarch64")))]
pub mod level1;

#[cfg(target_arch = "aarch64")]
#[cfg_attr(docsrs, doc(cfg(target_arch = "aarch64")))]
pub mod level2;

#[cfg(target_arch = "aarch64")]
#[cfg_attr(docsrs, doc(cfg(target_arch = "aarch64")))]
pub mod level3;

pub mod enums;

#[cfg(target_arch = "aarch64")]
#[cfg_attr(docsrs, doc(cfg(target_arch = "aarch64")))]
pub(crate) mod level1_special;


