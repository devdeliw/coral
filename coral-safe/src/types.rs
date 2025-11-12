use crate::errors::BufferError; 


/// Immutable Vector Type
#[derive(Debug, Copy, Clone)]
pub struct VectorRef<'a, T> { 
    data    : &'a [T], 
    n       : usize, 
    stride  : usize, 
    offset  : usize
}


/// Mutable Vector Type
#[derive(Debug)]
pub struct VectorMut<'a, T> { 
    data    : &'a mut [T], 
    n       : usize, 
    stride  : usize, 
    offset  : usize 
}


/// Immutable Matrix Type
#[derive(Debug, Copy, Clone)]
pub struct MatrixRef<'a, T> { 
    data        : &'a [T], 
    n_rows      : usize, 
    n_cols      : usize, 
    lda         : usize,
    offset      : usize
}


/// Mutable Matrix Type
#[derive(Debug)]
pub struct MatrixMut<'a, T> { 
    data        : &'a mut [T], 
    n_rows      : usize, 
    n_cols      : usize,
    lda         : usize,
    offset      : usize
}


impl<'a, T> VectorRef<'a, T> { 
    /// Constructor
    pub fn new ( 
        data    : &'a [T], 
        n       : usize, 
        stride  : usize, 
        offset  : usize
    ) -> Result<Self, BufferError> {
        if n == 0 { 
            return Ok( Self { 
                data, 
                n, 
                stride, 
                offset
            })
        } 

        if stride == 0 { 
            return Err(BufferError::ZeroStride);
        }

        let required_length = (n - 1)
            .saturating_mul(stride)
            .saturating_add(offset)
            .saturating_add(1); 
        let data_len = data.len(); 
        if required_length > data_len { 
            return Err(BufferError::OutOfBounds {
                required : required_length,
                len      : data_len 
            });
        }   
        
        Ok(Self { data, n, stride, offset })
    }

    /// Number of logical elements
    #[inline] pub fn n        (&self) -> usize   { self.n      } 
    /// Stride between logical elements
    #[inline] pub fn stride   (&self) -> usize   { self.stride }
    /// Offset to the first logical element
    #[inline] pub fn offset   (&self) -> usize   { self.offset }

    /// Returns full slice 
    #[inline] pub fn as_slice (&self) -> &[T] { self.data   }
    /// Returns immutable contiguous logical window
    #[inline] pub fn contiguous_slice (&self) -> Option<&[T]> { 
        (self.stride == 1).then(|| &self.data[self.offset..self.offset+self.n])
    }
}


impl<'a, T> VectorMut<'a, T> {
    /// Constructor
    pub fn new ( 
        data    : &'a mut [T], 
        n       : usize, 
        stride  : usize, 
        offset  : usize
    ) -> Result<Self, BufferError> { 
        if n == 0 { 
            return Ok( Self { 
                data, 
                n, 
                stride, 
                offset
            })
        } 

        if stride == 0 { 
            return Err(BufferError::ZeroStride);
        }

        let required_length = (n - 1)
            .saturating_mul(stride)
            .saturating_add(offset)
            .saturating_add(1); 

        let data_len = data.len(); 
        if required_length > data_len { 
            return Err(BufferError::OutOfBounds {
                required : required_length,
                len      : data_len 
            });
        }   
        
        Ok(Self { data, n, stride, offset })
    }

    /// Number of logical elements
    #[inline] pub fn n      (&self) -> usize { self.n      } 
    /// Stride between logical elements
    #[inline] pub fn stride (&self) -> usize { self.stride }
    /// Offset to the first logical element
    #[inline] pub fn offset (&self) -> usize { self.offset }

    /// Returns full slice 
    #[inline] pub fn as_slice (&self) -> &[T] { self.data }
    /// Returns full mutable slice 
    #[inline] pub fn as_mut_slice (&mut self) -> &mut [T] { self.data }

    /// Returns immutable contiguous logical window
    #[inline] pub fn contiguous_slice(&self) -> Option<&[T]> { 
        (self.stride == 1).then(|| &self.data[self.offset..self.offset+self.n])
    }
    /// Returns mutable contiguous logical window
    #[inline] pub fn contiguous_slice_mut (&mut self) -> Option<&mut [T]> { 
        (self.stride == 1).then(|| &mut self.data[self.offset..self.offset+self.n])
    }

}


impl<'a, T> MatrixRef<'a, T> {
    /// Constructor
    pub fn new ( 
        data    : &'a [T], 
        n_rows  : usize, 
        n_cols  : usize, 
        lda     : usize, 
        offset  : usize, 
    ) -> Result<Self, BufferError> { 
        if n_rows == 0 || n_cols == 0 { 
            return Ok( Self { 
                data, 
                n_rows, 
                n_cols, 
                lda, 
                offset
            })
        }

        if lda == 0 { 
            return Err(BufferError::ZeroStride); 
        }

        if lda < n_rows { 
            return Err(BufferError::InvalidLda { lda , n_rows });
        }

        let required_length = (n_cols - 1) 
            .saturating_mul(lda)
            .saturating_add(n_rows)
            .saturating_add(offset);
        let data_len = data.len();
        if required_length > data_len { 
            return Err(BufferError::OutOfBounds { 
                required : required_length, 
                len      : data_len 
            }); 
        }
        
        Ok( Self { data, n_rows, n_cols, lda, offset })
    }

    /// Number of rows
    #[inline] pub fn n_rows (&self) -> usize { self.n_rows }
    /// Number of columns
    #[inline] pub fn n_cols (&self) -> usize { self.n_cols } 
    /// Leading dimension
    #[inline] pub fn lda    (&self) -> usize { self.lda    } 
    /// Offset to the first logical element
    #[inline] pub fn offset (&self) -> usize { self.offset }

    /// Returns full slice 
    #[inline] pub fn as_slice (&self) -> &[T] { self.data   }

    /// Returns immutable contiguous logical window
    #[inline] pub fn contiguous_slice (&self) -> Option<&[T]> { 
        (self.lda == self.n_rows).then(
            || &self.data[
                self.offset..self.offset+(self.n_cols - 1)
                    .saturating_mul(self.lda) + 
                self.n_rows
            ]
        )
    }
}


impl<'a, T> MatrixMut<'a, T> { 
    /// Constructor
    pub fn new ( 
        data    : &'a mut [T], 
        n_rows  : usize, 
        n_cols  : usize, 
        lda     : usize, 
        offset  : usize, 
    ) -> Result<Self, BufferError> { 
        if n_rows == 0 || n_cols == 0 { 
            return Ok( Self { 
                data, 
                n_rows, 
                n_cols, 
                lda, 
                offset
            })
        }

        if lda == 0 { 
            return Err(BufferError::ZeroStride); 
        }

        if lda < n_rows { 
            return Err(BufferError::InvalidLda { lda, n_rows });
        }

        let required_length = (n_cols - 1) 
            .saturating_mul(lda)
            .saturating_add(n_rows)
            .saturating_add(offset);
        let data_len = data.len();
        if required_length > data_len { 
            return Err(BufferError::OutOfBounds { 
                required : required_length, 
                len      : data_len 
            }); 
        }
        
        Ok( Self { data, n_rows, n_cols, lda, offset })
    }

    /// Number of rows
    #[inline] pub fn n_rows (&self) -> usize { self.n_rows   }
    /// Number of columns
    #[inline] pub fn n_cols (&self) -> usize { self.n_cols   } 
    /// Leading dimension
    #[inline] pub fn lda    (&self) -> usize { self.lda      }
    /// Offset to the first logical element
    #[inline] pub fn offset (&self) -> usize { self.offset   }

    /// Returns full slice 
    #[inline] pub fn as_slice     (&self) -> &[T] { self.data }
    /// Returns full mutable slice 
    #[inline] pub fn as_mut_slice (&mut self) -> &mut [T] { self.data }

    /// Returns immutable contiguous logical window
    #[inline] pub fn contiguous_slice (&self) -> Option<&[T]> { 
        (self.lda == self.n_rows).then(
            || &self.data[
                self.offset..self.offset+(self.n_cols - 1)
                    .saturating_mul(self.lda) + 
                self.n_rows
            ]
        )
    }

    /// Returns mutable contiguous logical window
    #[inline] pub fn contiguous_slice_mut (&mut self) -> Option<&mut [T]> { 
        (self.lda == self.n_rows).then(
            || &mut self.data[
                self.offset..self.offset+(self.n_cols - 1)
                    .saturating_mul(self.lda) + 
                self.n_rows
            ]
        )
    }
}

