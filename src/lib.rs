#![doc = include_str!("../README.md")]

#[cfg(target_arch = "aarch64")]
pub mod level1;

#[cfg(target_arch = "aarch64")]
pub mod level2;

#[cfg(target_arch = "aarch64")]
pub mod level3;

pub mod enums;

#[cfg(target_arch = "aarch64")]
pub(crate) mod level1_special;


