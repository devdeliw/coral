use rand::thread_rng;
use rand::distributions::{Distribution, Standard}; 
use coral_safe::errors::BufferError;

pub(crate) type CoralResult = Result<(), BufferError>;

pub(crate) const RTOL: f32 = 1e-6; 
pub(crate) const ATOL: f32 = 1e-6; 


pub(crate) fn make_strided_vec (
    len: usize, 
    inc: usize
) -> Vec<f32> {

    if len == 0 { 
        return vec![1.0; 1]; 
    }

    let req_len = (len - 1) * inc + (len > 0) as usize;
    let mut buf = vec![0.0; req_len];

    let mut rng = thread_rng();
    let dist = Standard;
    for val in buf.iter_mut().step_by(inc).take(len) {
        *val = dist.sample(&mut rng);
    }

    buf
}

pub(crate) fn assert_close ( 
    a: &[f32], 
    b: &[f32], 
    rtol: f32, 
    atol: f32, 
) { 
    assert_eq!(a.len(), b.len()); 

    let iterator = a.iter().zip(b.iter()); 
    for (i, (&x, &y)) in iterator.enumerate() { 
        let delta = (x - y).abs(); 
        let tolerance = atol + rtol * x.abs().max(y.abs()); 

        assert!( 
            delta <= tolerance, 
            "mismatch at idx {i}: {x} vs. {y} (delta={delta}, tol={tolerance})"
        );  
    }
}
