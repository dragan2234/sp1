// use elliptic_curve::{AffinePoint, Curve, CurveArithmetic};
use crate::edwards;
use crate::edwards::EdwardsParameters;
use crate::AffinePoint;
use crate::EllipticCurve;
use num::BigUint;

pub fn biguint_to_bits_le(integer: &BigUint, num_bits: usize) -> Vec<bool> {
    let byte_vec = integer.to_bytes_le();
    let mut bits = Vec::new();
    for byte in byte_vec {
        for i in 0..8 {
            bits.push(byte & (1 << i) != 0);
        }
    }
    debug_assert!(bits.len() <= num_bits, "Number too large to fit in {num_bits} digits");
    bits.resize(num_bits, false);
    bits
}

pub fn biguint_to_limbs<const N: usize>(integer: &BigUint) -> [u8; N] {
    let mut bytes = integer.to_bytes_le();
    debug_assert!(bytes.len() <= N, "Number too large to fit in {N} limbs");
    bytes.resize(N, 0u8);
    let mut limbs = [0u8; N];
    limbs.copy_from_slice(&bytes);
    limbs
}

#[inline]
pub fn multi_scalar_mul<C: EdwardsParameters>(
    bases: &[AffinePoint<edwards::EdwardsCurve<C>>],
    scalars: &[BigUint],
) -> AffinePoint<edwards::EdwardsCurve<C>> {
    let mut result = AffinePoint::new(C::neutral().0, C::neutral().1);

    for (base, scalar) in bases.iter().zip(scalars.iter()) {
        // println!("base is: {:?}", base);
        // println!("scalar is: {:?}", scalar);

        // println!("scalar mul is: {:?}", base.scalar_mul(scalar));

        let baseing = base.scalar_mul(scalar);
        result = result.ed_add(&baseing);

        // println!("current result is: {:?}", result);
    }

    result
}

#[inline]
pub fn biguint_from_limbs(limbs: &[u8]) -> BigUint {
    BigUint::from_bytes_le(limbs)
}

cfg_if::cfg_if! {
    if #[cfg(feature = "bigint-rug")] {
        pub fn biguint_to_rug(integer: &BigUint) -> rug::Integer {
            let mut int = rug::Integer::new();
            unsafe {
                int.assign_bytes_radix_unchecked(integer.to_bytes_be().as_slice(), 256, false);
            }
            int
        }

        pub fn rug_to_biguint(integer: &rug::Integer) -> BigUint {
            let be_bytes = integer.to_digits::<u8>(rug::integer::Order::MsfBe);
            BigUint::from_bytes_be(&be_bytes)
        }
    }
}
