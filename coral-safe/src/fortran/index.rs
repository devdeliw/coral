/// Type to be used for BLAS indices/dimension arguments. 
/// Allows user to use LP64 [i32] or ILP64 [i64] 
pub trait BlasIdx: Copy { 
    fn to_usize(self) -> usize; 
}

impl BlasIdx for i32 { 
    #[inline] 
    fn to_usize(self) -> usize {
        assert!(self >= 0, "must be non-negative. coral works with positive strides only.");
        self as usize 
    }
}

impl BlasIdx for i64 { 
    #[inline] 
    fn to_usize(self) -> usize { 
        assert!(self >= 0, "must be non-negative. coral works with positive strides only."); 
        assert!(self <= usize::MAX as i64, "value does not fit in usize"); 
        self as usize 
    }
}
