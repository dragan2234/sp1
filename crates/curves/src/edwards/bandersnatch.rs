use std::str::FromStr;

use generic_array::GenericArray;
use num::{BigUint, Num, One};
use serde::{Deserialize, Serialize};
use typenum::{U32, U62};

use crate::{
    curve25519_dalek::CompressedEdwardsY,
    edwards::{EdwardsCurve, EdwardsParameters},
    params::{FieldParameters, NumLimbs},
    AffinePoint, CurveType, EllipticCurveParameters,
};

pub type Bandersnatch = EdwardsCurve<BandersnatchParameters>;

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BandersnatchParameters;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BandersnatchBaseField;

impl FieldParameters for BandersnatchBaseField {
    const MODULUS: &'static [u8] = &[
        115, 237, 167, 83, 41, 157, 125, 72, 51, 57, 216, 8, 9, 161, 216, 5, 83, 189, 164, 2, 255,
        254, 91, 254, 255, 255, 255, 255, 0, 0, 0, 1,
    ];

    const WITNESS_OFFSET: usize = 1usize << 14;

    fn modulus() -> BigUint {
        BigUint::from_bytes_be(&[
            115, 237, 167, 83, 41, 157, 125, 72, 51, 57, 216, 8, 9, 161, 216, 5, 83, 189, 164, 2,
            255, 254, 91, 254, 255, 255, 255, 255, 0, 0, 0, 1,
        ])
    }
}

impl NumLimbs for BandersnatchBaseField {
    type Limbs = U32;
    type Witness = U62;
}

impl EllipticCurveParameters for BandersnatchParameters {
    type BaseField = BandersnatchBaseField;
    const CURVE_TYPE: CurveType = CurveType::Bandersnatch;
}

impl EdwardsParameters for BandersnatchParameters {
    const D: GenericArray<u8, U32> = GenericArray::from_array([
        231, 88, 141, 24, 245, 242, 105, 179, 146, 79, 229, 119, 113, 103, 102, 203, 216, 182, 227,
        107, 248, 59, 110, 198, 203, 103, 194, 51, 38, 193, 137, 99,
    ]);

    const A: GenericArray<u8, U32> = GenericArray::from_array([
        252, 255, 255, 255, 254, 255, 255, 255, 254, 91, 254, 255, 2, 164, 189, 83, 5, 216, 161, 9,
        8, 216, 57, 51, 72, 125, 157, 41, 83, 167, 237, 115,
    ]);

    fn prime_group_order() -> BigUint {
        BigUint::from_str_radix(
            "13108968793781547619861935127046491459309155893440570251786403306729687672801",
            10,
        )
        .unwrap()
    }

    fn generator() -> (BigUint, BigUint) {
        let x = BigUint::from_str_radix(
            "18886178867200960497001835917649091219057080094937609519140440539760939937304",
            10,
        )
        .unwrap();
        let y = BigUint::from_str_radix(
            "19188667384257783945677642223292697773471335439753913231509108946878080696678",
            10,
        )
        .unwrap();
        (x, y)
    }
}

/// Computes the square root of a number in the base field of Bandersnatch.
///
/// This function always returns the nonnegative square root, in the sense that the least
/// significant bit of the result is always 0.
pub fn bandersnatch_sqrt(a: &BigUint) -> Option<BigUint> {
    // Here is a description of how to calculate sqrt in the Curve25519 base field:
    // ssh://git@github.com/succinctlabs/curve25519-dalek/blob/
    // e2d1bd10d6d772af07cac5c8161cd7655016af6d/curve25519-dalek/src/field.rs#L256

    let modulus = BandersnatchBaseField::modulus();
    // The exponent is (modulus+3)/8;
    let mut beta = a.modpow(
        &BigUint::from_str(
            "7237005577332262213973186563042994240829374041602535252466099000494570602494",
        )
        .unwrap(),
        &modulus,
    );

    // The square root of -1 in the field.
    // Take from here:
    // ssh://git@github.com/succinctlabs/curve25519-dalek/blob/
    // e2d1bd10d6d772af07cac5c8161cd7655016af6d/curve25519-dalek/src/backend/serial/u64/constants.
    // rs#L89
    let sqrt_m1 = BigUint::from_str(
        "19681161376707505956807079304988542015446066515923890162744021073123829784752",
    )
    .unwrap();

    let beta_squared = &beta * &beta % &modulus;
    let neg_a = &modulus - a;

    if beta_squared == neg_a {
        beta = (&beta * &sqrt_m1) % &modulus;
    }

    let correct_sign_sqrt = &beta_squared == a;
    let flipped_sign_sqrt = beta_squared == neg_a;

    if !correct_sign_sqrt && !flipped_sign_sqrt {
        return None;
    }

    let beta_bytes = beta.to_bytes_le();
    if (beta_bytes[0] & 1) == 1 {
        beta = (&modulus - &beta) % &modulus;
    }

    Some(beta)
}

pub fn decompress(compressed_point: &CompressedEdwardsY) -> Option<AffinePoint<Bandersnatch>> {
    let mut point_bytes = *compressed_point.as_bytes();
    let sign = point_bytes[31] >> 7 == 1;
    // mask out the sign bit
    point_bytes[31] &= 0b0111_1111;
    let modulus = &BandersnatchBaseField::modulus();

    let y = &BigUint::from_bytes_le(&point_bytes);
    let yy = &((y * y) % modulus);
    let u = (yy + modulus - BigUint::one()) % modulus; // u =  y²-1
    let v = &((yy * &BandersnatchParameters::d_biguint()) + &BigUint::one()) % modulus; // v = dy²+1

    let v_inv = v.modpow(&(modulus - BigUint::from(2u64)), modulus);
    let u_div_v = (u * &v_inv) % modulus;

    let mut x = bandersnatch_sqrt(&u_div_v)?;

    // sqrt always returns the nonnegative square root,
    // so we negate according to the supplied sign bit.
    if sign {
        x = (modulus - &x) % modulus;
    }

    Some(AffinePoint::new(x, y.clone()))
}

#[cfg(test)]
mod tests {

    use std::ops::Add;

    use super::*;
    use num::traits::ToBytes;

    const NUM_TEST_CASES: usize = 100;

    #[test]
    fn test_bandersnatch_program() {
        let mut point = {
            let (x, y) = BandersnatchParameters::generator();
            AffinePoint::<EdwardsCurve<BandersnatchParameters>>::new(x, y)
        };

        let double = point.clone().add(point.clone());

        let qwe = BigUint::from_str("80").unwrap();

        let scalar_mul_test = point.clone().scalar_mul(&qwe);
        // println!("affine double: {:?}", double);

        // println!("scalar mul test: {:?}", scalar_mul_test);
    }

    #[test]
    fn test_bandersnatch_decompress() {
        // This test checks that decompression of generator, 2x generator, 4x generator, etc. works.

        // Get the generator point.
        let mut point = {
            let (x, y) = BandersnatchParameters::generator();
            AffinePoint::<EdwardsCurve<BandersnatchParameters>>::new(x, y)
        };
        for _ in 0..NUM_TEST_CASES {
            // Compress the point. The first 255 bits of a compressed point is the y-coordinate. The
            // high bit of the 32nd byte gives the "sign" of x, which is the parity.
            let compressed_point = {
                let x = point.x.to_le_bytes();
                let y = point.y.to_le_bytes();
                let mut compressed = [0u8; 32];

                // Copy y into compressed.
                compressed[..y.len()].copy_from_slice(&y);

                // Set the sign bit.
                compressed[31] |= (x[0] & 1) << 7;

                CompressedEdwardsY(compressed)
            };
            assert_eq!(point, decompress(&compressed_point).unwrap());

            // Double the point to create a "random" point for the next iteration.
            point = point.clone() + point.clone();
        }
    }
}
