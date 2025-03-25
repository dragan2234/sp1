#![no_main]
sp1_zkvm::entrypoint!(main);
use std::str::FromStr;

use bytemuck::cast;
use sp1_curves::edwards::ed25519::Ed25519Parameters;
use sp1_curves::edwards::{bandersnatch::*, EdwardsCurve, EdwardsParameters};
use sp1_curves::{AffinePoint, BigUint, EllipticCurve};
use sp1_zkvm::lib::bandersnatch::*;
use sp1_zkvm::lib::utils::AffinePoint as Af;
use sp1_zkvm::syscalls::syscall_bandersnatch_add;

pub fn main() {
    // 18886178867200960497001835917649091219057080094937609519140440539760939937304
    // 19188667384257783945677642223292697773471335439753913231509108946878080696678
    let mut a: [u8; 64] = [
        24, 174, 82, 162, 102, 24, 231, 225, 101, 132, 153, 173, 34, 192, 121, 43, 243, 66, 190,
        123, 119, 17, 55, 116, 197, 52, 11, 44, 204, 50, 193, 41, 102, 65, 151, 204, 182, 103, 49,
        94, 96, 100, 228, 238, 129, 173, 140, 53, 134, 213, 220, 186, 80, 139, 125, 21, 15, 62, 18,
        218, 158, 102, 108, 42,
    ];
    // 21829743261194590194992413705867576097158323059182896808782966767024601242412
    // 19075870567762384361343718229920461045746972450262741916171739040424605531019
    let b: [u8; 64] = [
        44, 27, 99, 198, 199, 38, 54, 165, 237, 110, 2, 222, 169, 245, 62, 24, 126, 145, 194, 208,
        10, 239, 175, 217, 215, 119, 55, 185, 99, 50, 67, 48, 139, 59, 144, 24, 96, 2, 57, 16, 7,
        240, 101, 108, 127, 250, 13, 158, 130, 66, 43, 243, 133, 49, 238, 233, 238, 124, 136, 101,
        100, 143, 44, 42,
    ];
    println!("cycle-tracker-report-start: bandersnatch-add-syscall");
    syscall_bandersnatch_add(a.as_mut_ptr() as *mut [u32; 16], b.as_ptr() as *mut [u32; 16]);
    println!("cycle-tracker-report-end: bandersnatch-add-syscall");

    // 19213755708763254619264831853746015614457568707574289360541474768076689519718
    // 17364390373284516257285034247139577682165868767001357086426373468799918686336
    let c: [u8; 64] = [
        102, 52, 26, 98, 120, 34, 93, 101, 222, 174, 188, 209, 202, 65, 233, 31, 119, 89, 8, 5, 49,
        146, 75, 48, 68, 98, 10, 135, 176, 153, 122, 42, 128, 64, 0, 149, 254, 187, 101, 55, 44,
        150, 165, 46, 35, 137, 52, 181, 123, 20, 10, 112, 36, 149, 212, 132, 207, 167, 87, 193,
        139, 229, 99, 38,
    ];

    assert_eq!(a, c);

    let mut a_point = sp1_zkvm::lib::bandersnatch::Bandersnatch::new(cast(a.clone()));

    let b_point = sp1_zkvm::lib::bandersnatch::Bandersnatch::new(cast(b.clone()));
    println!("cycle-tracker-report-start: bandersnatch-add-syscall-inside");
    a_point.add_assign(&b_point);
    println!("cycle-tracker-report-end: bandersnatch-add-syscall-inside");
    let c_point = sp1_zkvm::lib::bandersnatch::Bandersnatch::new(cast(c.clone()));

    let scalar = 800;
    println!("cycle-tracker-report-start: bandersnatch-mul-syscall");
    a_point.mul_assign(&[800]);
    println!("cycle-tracker-report-end: bandersnatch-mul-syscall");

    assert_eq!(c_point.to_le_bytes().as_mut_slice(), c);

    let (x, y) = sp1_curves::edwards::bandersnatch::Bandersnatch::generator();

    let a_g: AffinePoint<EdwardsCurve<BandersnatchParameters>> = sp1_curves::AffinePoint::new(x, y);

    let (x_ed, y_ed) = sp1_curves::edwards::ed25519::Ed25519::generator();

    let a_g_ed: AffinePoint<EdwardsCurve<Ed25519Parameters>> =
        sp1_curves::AffinePoint::new(x_ed, y_ed);

    println!("cycle-tracker-report-start: bandersnatch-add-NO-syscall");
    let result = sp1_curves::edwards::bandersnatch::Bandersnatch::ec_add(&a_g, &a_g.clone());
    println!("cycle-tracker-report-end: bandersnatch-add-NO-syscall");

    println!("cycle-tracker-report-start: bandersnatch-mul-NO-syscall");
    a_g.scalar_mul(&BigUint::from_str("800").unwrap());
    println!("cycle-tracker-report-end: bandersnatch-mul-NO-syscall");

    let mut base_scalars_string = [
        "13108968793781547619861935127046491459309155893440570251786403306729687672800",
        "12108968793781547619861935127046491459309155893440570251786403306729687672799",
        "11108968793781547619861935127046491459309155893440570251786403306729687672798",
        "10108968793781547619861935127046491459309155893440570251786403306729687672797",
    ];

    let scalars_string: Vec<_> = base_scalars_string.iter().cycle().take(256).cloned().collect();

    let scalars: Vec<_> =
        scalars_string.clone().into_iter().map(|x| BigUint::from_str(x).unwrap()).collect();

    let mut base_points = [
        (
            "18886178867200960497001835917649091219057080094937609519140440539760939937304",
            "19188667384257783945677642223292697773471335439753913231509108946878080696678",
        ),
        (
            "21829743261194590194992413705867576097158323059182896808782966767024601242412",
            "19075870567762384361343718229920461045746972450262741916171739040424605531019",
        ),
        (
            "19213755708763254619264831853746015614457568707574289360541474768076689519718",
            "17364390373284516257285034247139577682165868767001357086426373468799918686336",
        ),
        (
            "9750030165270825669804718340526217750714550471942743324567662196134356926171",
            "17374475068068392136182018513238520664896609417860070833886908001325858074001",
        ),
    ];

    // Create an array of 256 elements by repeating `base_points`
    let points: Vec<_> = base_points.iter().cycle().take(256).cloned().collect();

    let mut affine_points: Vec<_> = points
        .into_iter()
        .map(|(x, y)| {
            AffinePoint::<EdwardsCurve<BandersnatchParameters>>::new(
                BigUint::from_str(x).unwrap(),
                BigUint::from_str(y).unwrap(),
            )
        })
        .collect();

    let afine_slice = affine_points.as_slice().clone();

    let mut scalars_slice = scalars;

    let points: Vec<sp1_zkvm::lib::bandersnatch::Bandersnatch> = affine_points
        .iter_mut()
        .map(|affine_point| {
            <sp1_zkvm::lib::bandersnatch::Bandersnatch as sp1_zkvm::lib::utils::AffinePoint<
                    16,
                >>::from(
                    affine_point.x.to_bytes_le().as_slice(),
                    affine_point.y.to_bytes_le().as_slice(),
                )
        })
        .collect();

    let scalars: Vec<_> =
        scalars_string.into_iter().map(|x| BigUint::from_str(x).unwrap().to_u32_digits()).collect();

    let scalar_refs: Vec<&[u32]> = scalars.iter().map(Vec::as_slice).collect();
    // sp1_zkvm::lib::utils::AffinePoint::from_le_bytes(bytes);
    // let points = affine_points;

    println!("cycle-tracker-report-start: msm-syscall");
    let precompile_msm = sp1_zkvm::lib::utils::AffinePoint::multi_scalar_multiplication_n(
        points.clone(),
        scalar_refs.clone(),
    );
    println!("cycle-tracker-report-end: msm-syscall");

    let precomputed_points = PrecomputedPoints::new(&points.clone(), 256);

    println!("cycle-tracker-report-start: precomputed-msm-syscall");
    let precompile_msm = sp1_zkvm::lib::bandersnatch::Bandersnatch::msm_with_precomputed(
        &scalar_refs.clone(),
        &precomputed_points,
    );
    println!("cycle-tracker-report-end: precomputed-msm-syscall");

    // assert_eq!(what, is);
}
