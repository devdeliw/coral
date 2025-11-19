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

/// Immutable Matrix Panel "view" 
/// Points to the first element in the view 
/// with metadata to parse over the view only
pub(crate) struct PanelRef<'a, T> { 
    mat     : &'a MatrixRef<'a, T>, 
    row_idx : usize, 
    col_idx : usize, 
    mr      : usize, 
    nr      : usize, 
}


/// Mutable Matrix Panel "view"
/// Points to the first element in the view 
/// with metadata to parse over the view only
pub(crate) struct PanelMut<'a, T> { 
    mat     : &'a mut MatrixMut<'a, T>, 
    row_idx : usize, 
    col_idx : usize, 
    mr      : usize, 
    nr      : usize, 
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

    #[inline] pub(crate) fn get_values (&self, beg: usize, end: usize) -> &[T] { 
        debug_assert!(end <= self.n, "last idx must be <= n"); 
        debug_assert!(beg <= end, "start idx must be before end idx"); 

        let data = self.as_slice(); 
        
        &data[beg .. end]
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

    #[inline] pub(crate) fn get_values (&self, beg: usize, end: usize) -> &[T] { 
        debug_assert!(end <= self.n, "last idx must be <= n"); 
        debug_assert!(beg <= end, "start idx must be before end idx"); 

        let data = self.as_slice(); 
        
        &data[beg .. end]
    }

    #[inline] pub(crate) fn get_values_mut (&mut self, beg: usize, end: usize) -> &mut [T] { 
        debug_assert!(end <= self.n, "last idx must be <= n"); 
        debug_assert!(beg <= end, "start idx must be before end idx"); 

        let data = self.as_slice_mut(); 
        
        &mut data[beg .. end]
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

    /// Returns the value of `Self[row_idx, col_idx]` 
    #[inline] pub(crate) fn at (&self, row_idx: usize, col_idx: usize) -> T { 
        self.data[col_idx * self.lda + row_idx]
    }

    /// Returns an immutable panel struct 
    #[inline] pub(crate) fn panel (
        &'a self, 
        row_idx: usize, 
        col_idx: usize, 
        mr: usize, 
        nr: usize, 
    ) -> PanelRef<'a, T> { 
        PanelRef { 
            mat: self, 
            row_idx, 
            col_idx, 
            mr, 
            nr
        }
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
    /// Returns the value of `Self[row_idx, col_idx]` 
    #[inline] pub(crate) fn at (&self, row_idx: usize, col_idx: usize) -> T { 
        self.data[col_idx * self.lda + row_idx]
    }

    /// Returns an mutable panel struct 
    #[inline] pub(crate) fn panel_mut (
        &'a mut self, 
        row_idx: usize, 
        col_idx: usize, 
        mr: usize, 
        nr: usize, 
    ) -> PanelMut<'a, T> { 
        PanelMut { 
            mat: self, 
            row_idx, 
            col_idx, 
            mr, 
            nr
        }
    }
}

impl<'a, T: Copy> PanelRef<'a, T> { 
    pub(crate) fn at(&self, i: usize, j: usize) -> T { 
        let row = self.row_idx + i; 
        let col = self.col_idx + j; 
        self.mat.at(row, col)
    }

    pub(crate) fn get_column_slice(&self, j: usize) -> &[T] { 
       debug_assert!(j <= self.nr, "col index must be < self.nr");  

        let lda = self.mat.lda(); 
        let off = self.mat.offset(); 
        let data = self.mat.as_slice(); 

        let col = self.col_idx + j; 
        let col_beg = off + self.row_idx + col * lda; 
        let col_end = col_beg + self.mr; 

        &data[col_beg .. col_end]
    }
}

impl<'a, T: Copy> PanelMut<'a, T> { 
    pub(crate) fn at(&self, i: usize, j: usize) -> T { 
        let row = self.row_idx + i; 
        let col = self.col_idx + j; 
        
        self.mat.at(row, col) 
    }

    pub(crate) fn get_column_slice(&mut self, j: usize) -> &mut [T] { 
        debug_assert!(j <= self.nr, "col index must be < self.nr");

        let lda = self.mat.lda(); 
        let off = self.mat.offset(); 
        let data = self.mat.as_slice_mut(); 

        let col = self.col_idx + j; 
        let col_beg = off + self.row_idx + col * lda; 
        let col_end = col_beg + self.mr; 

        &mut data[col_beg .. col_end]
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
