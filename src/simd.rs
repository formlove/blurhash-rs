pub unsafe fn branchless_max(mut x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        x = min;
    }
    if x > max {
        x = max;
    }
    x
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cmp::Ordering;

#[cfg(target_arch = "x86_64")]
union U8x16 {
    vector: __m128,
    bytes: [f32; 4],
}

#[cfg(target_arch = "x86_64")]
const ZERO: __m128 = unsafe {
    (U8x16 {
        bytes: [-0., -0., -0., -0.],
    })
    .vector
};

#[cfg(target_arch = "x86_64")]
pub unsafe fn find_abs_max(data: &[[f32; 4]]) -> f32 {
    let mut max = _mm_set1_ps(0.);

    for lane in data.iter() {
        let data = std::mem::transmute::<[f32; 4], __m128>(*lane);
        max = _mm_max_ps(max, _mm_andnot_ps(ZERO, data));
    }

    let res = std::mem::transmute::<__m128, [f32; 4]>(max);

    *res.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap()
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn find_abs_max(data: &[[f32; 4]]) -> f32 {
    data.into_iter()
        .map(|[a, b, c, _d]| f32::max(f32::max(f32::abs(*a), f32::abs(*b)), f32::abs(*c)))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use crate::simd::find_abs_max;

    #[test]
    fn test_max() {
        let data = vec![
            [0., 1., 21., 0.],
            [1., 8., 22., 0.],
            [2., 9., 23., 0.],
            [3., 10., 24., 0.],
            [4., 11., 25., 0.],
            [5., 12., 26., 0.],
            [6., 13., 27., 0.],
            [7., 14., 28., 0.],
            [7., 14., 28., 0.],
            [7., 14., 28., 0.],
            [7., 14., 28., 0.],
            [7., 14., 28., 0.],
            [7., 14., 28., 0.],
            [-500., 14., 28., 0.],
            [-500.1, 14., 28., 0.],
        ];

        assert_eq!(unsafe { find_abs_max(&data) }, 500.1)
    }
}
