use crate::errors::BufferError;

/// * [CoralTranspose::NoTrans] for no-transpose variants
/// * [CoralTranspose::Trans] for transpose variants
#[derive(Debug, Copy, Clone)]
pub enum CoralTranspose { 
    NoTrans, 
    Trans 
}

/// * [CoralTriangular::Upper] for upper-triangular variants
/// * [CoralTriangular::Lower] for lower-triangular variants
#[derive(Debug, Copy, Clone)]
pub enum CoralTriangular { 
    Upper, 
    Lower
}

/// * [CoralDiagonal::Unit] for unit diagonal variants
/// * [CoralDiagonal::NonUnit] for non-unit diagonal variants
#[derive(Debug, Copy, Clone)]
pub enum CoralDiagonal { 
    Unit, 
    NonUnit, 
}


impl CoralDiagonal { 
    pub fn is_unit ( &self ) -> bool { 
        match self { 
            CoralDiagonal::Unit    => true, 
            CoralDiagonal::NonUnit => false, 
        }
    }
}

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

    /// Checks whether the number `n` of logical elements is equal to a value
    /// Used for asserting two Vector types have an equal `n` elements to parse
    #[inline] pub fn compare_n (&self, n: usize) -> bool { 
        self.n == n
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
    #[inline] pub fn as_slice_mut (&mut self) -> &mut [T] { self.data }

    /// Returns immutable contiguous logical window
    #[inline] pub fn contiguous_slice(&self) -> Option<&[T]> { 
        (self.stride == 1).then(|| &self.data[self.offset..self.offset+self.n])
    }
    /// Returns mutable contiguous logical window
    #[inline] pub fn contiguous_slice_mut (&mut self) -> Option<&mut [T]> { 
        (self.stride == 1).then(|| &mut self.data[self.offset..self.offset+self.n])
    }

    /// Checks whether the number `n` of logical elements is equal to a value
    /// Used for asserting two Vector types have an equal `n` elements to parse
    #[inline] pub fn compare_n (&self, n: usize) -> bool { 
        self.n == n
    }
}

impl<'a, T: Copy> MatrixRef<'a, T> {
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

    /// Checks whether `self.n_cols == self.n_rows` 
    /// for square matrices 
    #[inline] pub fn compare_m_n (&self) -> bool { 
        self.n_rows == self.n_cols
    }
}

impl<'a, T: Copy> MatrixMut<'a, T> { 
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
    #[inline] pub fn as_slice_mut (&mut self) -> &mut [T] { self.data }

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

    /// Checks whether `self.n_cols == self.n_rows` 
    /// for square matrices 
    #[inline] pub fn compare_m_n (&self) -> bool { 
        self.n_rows == self.n_cols
    }
}

/// Used to assert two any Vector have the same 
/// number of logical elements to access 
#[macro_export]
macro_rules! debug_assert_n_eq {
    ($x:expr, $y:expr) => {
        debug_assert!(
            $x.compare_n($y.n()),
            "number of logical elements must be equal"
        );
    };
}
