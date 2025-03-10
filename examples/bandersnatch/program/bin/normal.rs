#![no_main]
sp1_zkvm::entrypoint!(main);
use std::io::Read;

use sp1_zkvm::lib::bandersnatch::*;
use sp1_zkvm::lib::utils::AffinePoint;
use sp1_zkvm::syscalls::syscall_bandersnatch_add;

#[sp1_derive::cycle_tracker]
pub fn expensive_function(x: usize) -> usize {
    let mut y = 1;
    for _ in 0..1000 {
        y *= x;
        y %= 7919;
    }

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

    syscall_bandersnatch_add(a.as_mut_ptr() as *mut [u32; 16], b.as_ptr() as *mut [u32; 16]);

    // 19213755708763254619264831853746015614457568707574289360541474768076689519718
    // 17364390373284516257285034247139577682165868767001357086426373468799918686336
    let c: [u8; 64] = [
        102, 52, 26, 98, 120, 34, 93, 101, 222, 174, 188, 209, 202, 65, 233, 31, 119, 89, 8, 5, 49,
        146, 75, 48, 68, 98, 10, 135, 176, 153, 122, 42, 128, 64, 0, 149, 254, 187, 101, 55, 44,
        150, 165, 46, 35, 137, 52, 181, 123, 20, 10, 112, 36, 149, 212, 132, 207, 167, 87, 193,
        139, 229, 99, 38,
    ];

    assert_eq!(a, c);

    let mut a_point = Bandersnatch::from_le_bytes(a.clone().as_mut_slice());

    let b_point = Bandersnatch::from_le_bytes(b.clone().as_mut_slice());

    a_point.add_assign(&b_point);

    let c_point = Bandersnatch::from_le_bytes(c.clone().as_mut_slice());

    assert_eq!(c_point.to_le_bytes().as_mut_slice(), c);

    println!("dones");

    y
}

pub fn main() {
    let mut nums = vec![1, 1];

    // Setup a large vector with Fibonacci-esque numbers.
    println!("cycle-tracker-start: setup");
    for _ in 0..100 {
        let mut c = nums[nums.len() - 1] + nums[nums.len() - 2];
        c %= 7919;
        nums.push(c);
    }
    println!("cycle-tracker-end: setup");

    println!("cycle-tracker-start: main-body");
    for i in 0..2 {
        let result = expensive_function(nums[nums.len() - i - 1]);
        println!("result: {}", result);
    }
    println!("cycle-tracker-end: main-body");
}
