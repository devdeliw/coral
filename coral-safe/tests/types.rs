use coral_safe::types::{MatrixRef, MatrixMut, VectorMut, VectorRef}; 
use coral_safe::errors::BufferError;  

type CoralResult = Result<(), BufferError>;

#[test] 
fn ensure_nonzero_stride_vec() -> CoralResult { 
    let n = 5; 
    let stride = 0; 
    let data_ref = vec![1.0; n]; 
    let mut data_mut = vec![1.0; n]; 
    let offset = 0; 

    let coral_vec_ref = VectorRef::new ( 
        &data_ref, 
        n, 
        stride, 
        offset, 
    );

    let coral_vec_mut = VectorMut::new ( 
        &mut data_mut, 
        n, 
        stride,
        offset, 
    ); 

    assert_eq!(coral_vec_ref.unwrap_err(), BufferError::ZeroStride);
    assert_eq!(coral_vec_mut.unwrap_err(), BufferError::ZeroStride);
    Ok(())
}

#[test]
fn ensure_within_bounds_vec() -> CoralResult { 
    let n = 3; 
    let stride = 2; 
    let data_fail_ref = vec![1.0; 5]; 
    let data_pass_ref = vec![1.0; 6];
    let data_fail_mut = vec![1.0; 5]; 
    let data_pass_mut = vec![1.0; 6];
    let offset = 1; 

    let coral_vec_fail_ref = VectorRef::new ( 
        &data_fail_ref, 
        n, 
        stride,
        offset, 
    ); 

    let coral_vec_pass_ref = VectorRef::new ( 
        &data_pass_ref, 
        n, 
        stride, 
        offset, 
    );

    let coral_vec_fail_mut = VectorRef::new ( 
        &data_fail_mut, 
        n, 
        stride,
        offset, 
    ); 

    let coral_vec_pass_mut = VectorRef::new ( 
        &data_pass_mut, 
        n, 
        stride, 
        offset, 
    );

    assert_eq!(coral_vec_fail_ref.unwrap_err(), BufferError::OutOfBounds { required: 6, len: 5 });
    assert_eq!(coral_vec_fail_mut.unwrap_err(), BufferError::OutOfBounds { required: 6, len: 5 }); 
    assert!(coral_vec_pass_ref.is_ok()); 
    assert!(coral_vec_pass_mut.is_ok()); 
    Ok(())
}


#[test]
fn ensure_valid_lda_mat() -> CoralResult { 
    let n_rows = 5; 
    let n_cols = 6; 

    let lda_valid   = 7; 
    let lda_invalid = 4;

    let abuf_valid_ref = vec![1.0; n_cols * lda_valid]; 
    let mut abuf_valid_mut = vec![1.0; n_cols * lda_valid]; 

    let abuf_invalid_ref = vec![1.0; n_cols * lda_invalid]; 
    let mut abuf_invalid_mut = vec![1.0; n_cols * lda_invalid]; 


    let coral_mat_pass_ref = MatrixRef::new ( 
        &abuf_valid_ref, 
        n_rows, 
        n_cols, 
        lda_valid, 
        0, 
    );

    let coral_mat_fail_ref = MatrixRef::new ( 
        &abuf_invalid_ref, 
        n_rows, 
        n_cols, 
        lda_invalid, 
        0, 
    ); 

    let coral_mat_pass_mut = MatrixMut::new ( 
        &mut abuf_valid_mut, 
        n_rows, 
        n_cols, 
        lda_valid, 
        0, 
    );

    let coral_mat_fail_mut = MatrixMut::new ( 
        &mut abuf_invalid_mut, 
        n_rows, 
        n_cols, 
        lda_invalid, 
        0, 
    ); 

    assert_eq!(coral_mat_fail_ref.unwrap_err(), BufferError::InvalidLda { lda: lda_invalid, n_rows });
    assert_eq!(coral_mat_fail_mut.unwrap_err(), BufferError::InvalidLda { lda: lda_invalid, n_rows });
    assert!(coral_mat_pass_ref.is_ok());
    assert!(coral_mat_pass_mut.is_ok());
    Ok(())
}

