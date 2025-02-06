#![no_main]

use sp1_zkvm::syscalls::syscall_bandersnatch_add;

sp1_zkvm::entrypoint!(main);

pub fn main() {
    for _ in 0..4 {
        // 18886178867200960497001835917649091219057080094937609519140440539760939937304
        // 19188667384257783945677642223292697773471335439753913231509108946878080696678
        let mut a: [u8; 64] = [
            232, 66, 76, 80, 103, 12, 14, 133, 197, 159, 68, 238, 152, 18, 205, 43, 164, 148, 233,
            19, 92, 209, 206, 111, 110, 137, 164, 9, 22, 248, 215, 41, 134, 15, 43, 189, 144, 181,
            193, 209, 185, 233, 78, 94, 193, 7, 88, 215, 217, 15, 56, 155, 8, 28, 185, 27, 156,
            236, 229, 112, 47, 217, 124, 42,
        ];

        // 18886178867200960497001835917649091219057080094937609519140440539760939937304
        // 19188667384257783945677642223292697773471335439753913231509108946878080696678
        let b: [u8; 64] = [
            232, 66, 76, 80, 103, 12, 14, 133, 197, 159, 68, 238, 152, 18, 205, 43, 164, 148, 233,
            19, 92, 209, 206, 111, 110, 137, 164, 9, 22, 248, 215, 41, 134, 15, 43, 189, 144, 181,
            193, 209, 185, 233, 78, 94, 193, 7, 88, 215, 217, 15, 56, 155, 8, 28, 185, 27, 156,
            236, 229, 112, 47, 217, 124, 42,
        ];

        syscall_bandersnatch_add(a.as_mut_ptr() as *mut [u32; 16], b.as_ptr() as *mut [u32; 16]);

        // 21829743261194590194992413705867576097158323059182896808782966767024601242412
        // 19075870567762384361343718229920461045746972450262741916171739040424605531019
        let c: [u8; 64] = [
            12, 126, 246, 6, 40, 123, 231, 57, 181, 178, 143, 180, 20, 155, 111, 200, 200, 191,
            137, 91, 152, 231, 72, 110, 46, 137, 93, 119, 143, 223, 88, 48, 219, 51, 137, 71, 54,
            92, 145, 42, 43, 166, 96, 189, 119, 98, 239, 16, 219, 176, 199, 21, 152, 78, 64, 170,
            52, 156, 88, 84, 226, 251, 60, 42,
        ];

        assert_eq!(a, c);
    }

    println!("done");
}
