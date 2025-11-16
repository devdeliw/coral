use rand::thread_rng;
use rand::distributions::{Distribution, Standard}; 
use coral_safe::types::{VectorMut, VectorRef};

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

#[inline]
pub(crate) fn bytes(n: usize, alpha: usize) -> u64 { 
    (alpha * n * std::mem::size_of::<f32>()) as u64
}

#[inline] 
pub(crate) fn make_view_ref<'a>(x: &'a[f32], n: usize, incx: usize) -> VectorRef<'a, f32> { 
    VectorRef::new(x, n, incx, 0).expect("x view ref")
}

#[inline] 
pub(crate) fn make_view_mut<'a>(x: &'a mut [f32], n: usize, incx: usize) -> VectorMut<'a, f32> { 
    VectorMut::new(x, n, incx, 0).expect("x view ref")
}


