use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use std::{fmt::Debug, marker::PhantomData};

use hashbrown::HashMap;
use itertools::Itertools;
use num::{BigUint, Zero};

use crate::air::MemoryAirBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord, EllipticCurveAddEvent, FieldOperation, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_curves::{
    edwards::{
        bandersnatch::BandersnatchBaseField, EdwardsParameters, NUM_LIMBS, WORDS_CURVE_POINT,
    },
    params::{FieldParameters, Limbs, NumLimbs},
    AffinePoint, EllipticCurve,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::air::{BaseAirBuilder, InteractionScope, MachineAir, SP1AirBuilder};

use crate::{
    memory::{value_as_limbs, MemoryReadCols, MemoryWriteCols},
    operations::field::{
        field_den::FieldDenCols, field_inner_product::FieldInnerProductCols, field_op::FieldOpCols,
        range::FieldLtCols,
    },
    utils::{limbs_from_prev_access, pad_rows_fixed},
};

pub const NUM_BANDERSNATCH_ADD_COLS: usize = size_of::<BandersnatchAddAssignCols<u8>>();

/// A set of columns to compute `BandersnatchAdd` where a, b are field elements.
/// Right now the number of limbs is assumed to be a constant, although this could be macro-ed
/// or made generic in the future.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct BandersnatchAddAssignCols<T> {
    pub is_real: T,
    pub shard: T,
    pub clk: T,
    pub p_ptr: T,
    pub q_ptr: T,
    pub p_access: [MemoryWriteCols<T>; WORDS_CURVE_POINT],
    pub q_access: [MemoryReadCols<T>; WORDS_CURVE_POINT],
    pub(crate) x3_numerator: FieldInnerProductCols<T, BandersnatchBaseField>,
    pub(crate) y3_numerator_with_a: FieldOpCols<T, BandersnatchBaseField>,
    pub(crate) y3_numerator: FieldInnerProductCols<T, BandersnatchBaseField>,
    pub(crate) x1_mul_y1: FieldOpCols<T, BandersnatchBaseField>,
    pub(crate) x2_mul_y2: FieldOpCols<T, BandersnatchBaseField>,
    pub(crate) f: FieldOpCols<T, BandersnatchBaseField>,
    pub(crate) d_mul_f: FieldOpCols<T, BandersnatchBaseField>,
    pub(crate) x3_ins: FieldDenCols<T, BandersnatchBaseField>,
    pub(crate) y3_ins: FieldDenCols<T, BandersnatchBaseField>,
    pub(crate) x3_range: FieldLtCols<T, BandersnatchBaseField>,
    pub(crate) y3_range: FieldLtCols<T, BandersnatchBaseField>,
}

#[derive(Default)]
pub struct BandersnatchAddAssignChip<E> {
    _marker: PhantomData<E>,
}

impl<E: EllipticCurve + EdwardsParameters> BandersnatchAddAssignChip<E> {
    pub const fn new() -> Self {
        Self { _marker: PhantomData }
    }

    #[allow(clippy::too_many_arguments)]
    fn populate_field_ops<F: PrimeField32>(
        record: &mut impl ByteRecord,
        cols: &mut BandersnatchAddAssignCols<F>,
        p_x: BigUint,
        p_y: BigUint,
        q_x: BigUint,
        q_y: BigUint,
    ) {
        let x3_numerator = cols.x3_numerator.populate(
            record,
            &[p_x.clone(), q_x.clone()],
            &[q_y.clone(), p_y.clone()],
        );

        let minus_a = E::BaseField::modulus() - E::a_biguint();

        let y3_numerator_with_a =
            cols.y3_numerator_with_a.populate(record, &minus_a, &p_x, FieldOperation::Mul);

        let y3_numerator = cols.y3_numerator.populate(
            record,
            &[p_y.clone(), y3_numerator_with_a.clone()],
            &[q_y.clone(), q_x.clone()],
        );

        let x1_mul_y1 = cols.x1_mul_y1.populate(record, &p_x, &p_y, FieldOperation::Mul);
        let x2_mul_y2 = cols.x2_mul_y2.populate(record, &q_x, &q_y, FieldOperation::Mul);
        let f = cols.f.populate(record, &x1_mul_y1, &x2_mul_y2, FieldOperation::Mul);

        let d = E::d_biguint();
        let d_mul_f = cols.d_mul_f.populate(record, &f, &d, FieldOperation::Mul);

        let x3 = cols.x3_ins.populate(record, &x3_numerator, &d_mul_f, true);
        let y3 = cols.y3_ins.populate(record, &y3_numerator, &d_mul_f, false);

        cols.x3_range.populate(record, &x3, &BandersnatchBaseField::modulus());
        cols.y3_range.populate(record, &y3, &BandersnatchBaseField::modulus());
    }
}

impl<F: PrimeField32, E: EllipticCurve + EdwardsParameters> MachineAir<F>
    for BandersnatchAddAssignChip<E>
{
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "BandersnatchAddAssign".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let events = input.get_precompile_events(SyscallCode::ED_ADD);

        let mut rows = events
            .par_iter()
            .map(|(_, event)| {
                let event = if let PrecompileEvent::BandersnatchAdd(event) = event {
                    event
                } else {
                    unreachable!();
                };

                let mut row = [F::zero(); NUM_BANDERSNATCH_ADD_COLS];
                let cols: &mut BandersnatchAddAssignCols<F> = row.as_mut_slice().borrow_mut();
                let mut blu = Vec::new();
                self.event_to_row(event, cols, &mut blu);
                row
            })
            .collect::<Vec<_>>();

        pad_rows_fixed(
            &mut rows,
            || {
                let mut row = [F::zero(); NUM_BANDERSNATCH_ADD_COLS];
                let cols: &mut BandersnatchAddAssignCols<F> = row.as_mut_slice().borrow_mut();
                let zero = BigUint::zero();
                Self::populate_field_ops(
                    &mut vec![],
                    cols,
                    zero.clone(),
                    zero.clone(),
                    zero.clone(),
                    zero,
                );
                row
            },
            input.fixed_log2_rows::<F, _>(self),
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_BANDERSNATCH_ADD_COLS,
        )
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let events = input.get_precompile_events(SyscallCode::ED_ADD);
        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(_, event)| {
                    let event = if let PrecompileEvent::BandersnatchAdd(event) = event {
                        event
                    } else {
                        unreachable!();
                    };

                    let mut row = [F::zero(); NUM_BANDERSNATCH_ADD_COLS];
                    let cols: &mut BandersnatchAddAssignCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::ED_ADD).is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<E: EllipticCurve + EdwardsParameters> BandersnatchAddAssignChip<E> {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &EllipticCurveAddEvent,
        cols: &mut BandersnatchAddAssignCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        // Decode affine points.
        let p = &event.p;
        let q = &event.q;
        let p = AffinePoint::<E>::from_words_le(p);
        let (p_x, p_y) = (p.x, p.y);
        let q = AffinePoint::<E>::from_words_le(q);
        let (q_x, q_y) = (q.x, q.y);

        // Populate basic columns.
        cols.is_real = F::one();
        cols.shard = F::from_canonical_u32(event.shard);
        cols.clk = F::from_canonical_u32(event.clk);
        cols.p_ptr = F::from_canonical_u32(event.p_ptr);
        cols.q_ptr = F::from_canonical_u32(event.q_ptr);

        Self::populate_field_ops(blu, cols, p_x, p_y, q_x, q_y);

        // Populate the memory access columns.
        for i in 0..WORDS_CURVE_POINT {
            cols.q_access[i].populate(event.q_memory_records[i], blu);
        }
        for i in 0..WORDS_CURVE_POINT {
            cols.p_access[i].populate(event.p_memory_records[i], blu);
        }
    }
}

impl<F, E: EllipticCurve + EdwardsParameters> BaseAir<F> for BandersnatchAddAssignChip<E> {
    fn width(&self) -> usize {
        NUM_BANDERSNATCH_ADD_COLS
    }
}

impl<AB, E: EllipticCurve + EdwardsParameters> Air<AB> for BandersnatchAddAssignChip<E>
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BandersnatchAddAssignCols<AB::Var> = (*local).borrow();

        let x1: Limbs<AB::Var, <BandersnatchBaseField as NumLimbs>::Limbs> =
            limbs_from_prev_access(&local.p_access[0..8]);
        let x2: Limbs<AB::Var, <BandersnatchBaseField as NumLimbs>::Limbs> =
            limbs_from_prev_access(&local.q_access[0..8]);
        let y1: Limbs<AB::Var, <BandersnatchBaseField as NumLimbs>::Limbs> =
            limbs_from_prev_access(&local.p_access[8..16]);
        let y2: Limbs<AB::Var, <BandersnatchBaseField as NumLimbs>::Limbs> =
            limbs_from_prev_access(&local.q_access[8..16]);

        // x3_numerator = x1 * y2 + x2 * y1.
        local.x3_numerator.eval(builder, &[x1, x2], &[y2, y1], local.is_real);

        let a_minus_biguint = E::BaseField::modulus() - E::a_biguint();
        let a_const = E::BaseField::to_limbs_field::<AB::Expr, _>(&a_minus_biguint);
        local.y3_numerator_with_a.eval(builder, &a_const, &x1, FieldOperation::Mul, local.is_real);

        // y3_numerator = y1 * y2 + x1 * x2.
        local.y3_numerator.eval(builder, &[y1, x1], &[y2, x2], local.is_real);

        // f = x1 * x2 * y1 * y2.
        local.x1_mul_y1.eval(builder, &x1, &y1, FieldOperation::Mul, local.is_real);
        local.x2_mul_y2.eval(builder, &x2, &y2, FieldOperation::Mul, local.is_real);

        let x1_mul_y1 = local.x1_mul_y1.result;
        let x2_mul_y2 = local.x2_mul_y2.result;
        local.f.eval(builder, &x1_mul_y1, &x2_mul_y2, FieldOperation::Mul, local.is_real);

        // d * f.
        let f = local.f.result;
        let d_biguint = E::d_biguint();
        let d_const = E::BaseField::to_limbs_field::<AB::Expr, _>(&d_biguint);
        local.d_mul_f.eval(builder, &f, &d_const, FieldOperation::Mul, local.is_real);

        let d_mul_f = local.d_mul_f.result;

        let modulus = BandersnatchBaseField::to_limbs_field::<AB::Expr, AB::F>(
            &BandersnatchBaseField::modulus(),
        );

        // x3 = x3_numerator / (1 + d * f).
        local.x3_ins.eval(builder, &local.x3_numerator.result, &d_mul_f, true, local.is_real);
        local.x3_range.eval(builder, &local.x3_ins.result, &modulus, local.is_real);

        // y3 = y3_numerator / (1 - d * f).
        local.y3_ins.eval(builder, &local.y3_numerator.result, &d_mul_f, false, local.is_real);
        local.y3_range.eval(builder, &local.y3_ins.result, &modulus, local.is_real);

        // Constraint self.p_access.value = [self.x3_ins.result, self.y3_ins.result]
        // This is to ensure that p_access is updated with the new value.
        let p_access_vec = value_as_limbs(&local.p_access);
        builder
            .when(local.is_real)
            .assert_all_eq(local.x3_ins.result, p_access_vec[0..NUM_LIMBS].to_vec());
        builder
            .when(local.is_real)
            .assert_all_eq(local.y3_ins.result, p_access_vec[NUM_LIMBS..NUM_LIMBS * 2].to_vec());

        builder.eval_memory_access_slice(
            local.shard,
            local.clk.into(),
            local.q_ptr,
            &local.q_access,
            local.is_real,
        );

        builder.eval_memory_access_slice(
            local.shard,
            local.clk + AB::F::from_canonical_u32(1),
            local.p_ptr,
            &local.p_access,
            local.is_real,
        );

        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_canonical_u32(SyscallCode::ED_ADD.syscall_id()),
            local.p_ptr,
            local.q_ptr,
            local.is_real,
            InteractionScope::Local,
        );
    }
}

#[cfg(test)]
mod tests {
    use sp1_core_executor::Program;
    use sp1_stark::CpuProver;
    use test_artifacts::BANDERSNATCH_ADD_ELF;

    use crate::{io::SP1Stdin, utils};

    #[test]
    fn test_bandersnatch_add_simple() {
        utils::setup_logger();
        let program = Program::from(BANDERSNATCH_ADD_ELF).unwrap();
        let stdin = SP1Stdin::new();
        utils::run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    }
}
