use core::fmt; 

#[derive(Debug, PartialEq, Eq)]
pub enum BufferError { 
    ZeroStride, 
    OutOfBounds { required: usize, len: usize }, 
    BadOffset   { offset: usize, len: usize   }, 
    InvalidLda  { lda: usize, n_rows: usize   }, 
}

impl fmt::Display for BufferError { 
    fn fmt (&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self { 
            BufferError::ZeroStride => 
                write!(f, "stride must be nonzero"), 
            BufferError::OutOfBounds { required, len } => 
                write!(f, "buffer too small: required {required}, have {len}"), 
            BufferError::BadOffset { offset, len } => 
                write!(f, "offset {offset} out of range for buffer of len {len}"), 
            BufferError::InvalidLda { lda, n_rows } => 
                write!(f, "leading dimension {lda} must exceed num rows {n_rows}")
        }
    }
}
impl std::error::Error for BufferError {} 
