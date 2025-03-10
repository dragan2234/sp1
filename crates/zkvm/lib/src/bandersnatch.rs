use crate::{syscall_bandersnatch_add, utils::AffinePoint};
use std::collections::HashMap;

/// The number of limbs in [Bandersnatch].
pub const N: usize = 16;

/// An affine point on the Bandersnatch curve.
#[derive(Copy, Clone)]
#[repr(align(4))]
pub struct Bandersnatch(pub [u32; N]);

impl AffinePoint<N> for Bandersnatch {
    /// The generator/base point for the Bandersnatch curve.
    const GENERATOR: [u32; N] = [
        404820167, 3008044021, 2006128210, 3415188337, 1811506904, 3322195704, 866396171,
        1676046374, 3425725798, 1595339185, 3987094881, 891267660, 1699467334, 904269297,
        2353212446, 712841230,
    ];

    #[allow(deprecated)]
    const GENERATOR_T: Self = Self(Self::GENERATOR);

    fn new(limbs: [u32; N]) -> Self {
        Self(limbs)
    }

    fn identity() -> Self {
        Self::identity()
    }

    fn is_identity(&self) -> bool {
        self.0 == Self::IDENTITY
    }

    fn limbs_ref(&self) -> &[u32; N] {
        &self.0
    }

    fn limbs_mut(&mut self) -> &mut [u32; N] {
        &mut self.0
    }

    fn add_assign(&mut self, other: &Self) {
        let a = self.limbs_mut();
        let b = other.limbs_ref();
        unsafe {
            syscall_bandersnatch_add(a, b);
        }
    }

    /// In Edwards curves, doubling is the same as adding a point to itself.
    fn double(&mut self) {
        let a = self.limbs_mut();
        unsafe {
            syscall_bandersnatch_add(a, a);
        }
    }

    /// Performs multi-scalar multiplication (MSM) using `mul_assign`.
    /// Scalars must be in little-endian `&[u32]` format.
    fn multi_scalar_multiplication_n(points: Vec<Self>, scalars: Vec<&[u32]>) -> Self {
        let mut res = Self::identity();

        for (point, scalar) in points.iter().zip(scalars.iter()) {
            let mut temp_point = point.clone();
            temp_point.mul_assign(scalar); // Efficient scalar multiplication
            res.add_assign(&temp_point); // Accumulate the result
        }

        res
    }
}

impl Bandersnatch {
    const IDENTITY: [u32; N] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];

    pub fn identity() -> Self {
        Self(Self::IDENTITY)
    }

    pub fn msm_with_precomputed(scalars: &[&[u32]], precomputed: &PrecomputedPoints) -> Self {
        let mut res = Self::identity();

        for (i, scalar) in scalars.iter().enumerate() {
            let mut temp_res = Self::identity();

            for (bit_idx, &word) in scalar.iter().enumerate() {
                for j in 0..32 {
                    if (word >> j) & 1 == 1 {
                        temp_res.add_assign(&precomputed.multiples[i][bit_idx * 32 + j]);
                    }
                }
            }

            res.add_assign(&temp_res);
        }

        res
    }
}

pub struct PrecomputedPoints {
    multiples: Vec<Vec<Bandersnatch>>, // multiples[i] stores multiples of points[i]
}

impl PrecomputedPoints {
    pub fn new(points: &[Bandersnatch], max_bits: usize) -> Self {
        let mut multiples = Vec::new();

        for point in points {
            let mut point_multiples = vec![point.clone()];

            let mut temp = point.clone();
            for _ in 1..max_bits {
                temp.double();
                point_multiples.push(temp.clone());
            }

            multiples.push(point_multiples);
        }

        Self { multiples }
    }
}
