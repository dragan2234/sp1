use crate::{syscall_bandersnatch_add, utils::AffinePoint};

/// The number of limbs in [Bandersnatch].
pub const N: usize = 16;

/// An affine point on the Bandersnatch curve.
#[derive(Copy, Clone)]
#[repr(align(4))]
pub struct Bandersnatch(pub [u32; N]);

impl AffinePoint<N> for Bandersnatch {
    /// The generator/base point for the Bandersnatch curve. Reference: https://datatracker.ietf.org/doc/html/rfc7748#section-4.1
    const GENERATOR: [u32; N] = [
        404820167, 3008044021, 2006128210, 3415188337, 1811506904, 3322195704, 866396171,
        1676046374, 3425725798, 1595339185, 3987094881, 891267660, 1699467334, 904269297,
        2353212446, 712841230,
    ];

    fn new(limbs: [u32; N]) -> Self {
        Self(limbs)
    }

    fn identity() -> Self {
        Self::identity()
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
}

impl Bandersnatch {
    const IDENTITY: [u32; N] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];

    pub fn identity() -> Self {
        Self(Self::IDENTITY)
    }
}
