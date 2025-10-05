use ark_ff::{Field, Zero};
use ark_pallas::Fr as F;
use utils::fr_from_hash;

// Poseidon permutation parameters for benchmarking and M1 scaffolding.
// Width t=17 matches Merkle arity m=16 with capacity c=1.
pub const T: usize = 17;       // state width = rate(16) + capacity(1)
pub const RATE: usize = 16;
pub const CAPACITY: usize = 1;
pub const RF: usize = 8;       // number of full rounds (total)
pub const RP: usize = 64;      // number of partial rounds
pub const ALPHA: u64 = 5;      // S-box x^5

#[derive(Clone)]
pub struct PoseidonParams {
    pub mds: [[F; T]; T],
    pub rc_full: [[F; T]; RF],
    pub rc_partial: [F; RP],
}

#[inline]
pub fn sbox5(x: F) -> F {
    // x^5 = x * x^2 * x^2
    let x2 = x.square();
    let x4 = x2.square();
    x * x4
}

pub fn permute(state: &mut [F; T], params: &PoseidonParams) {
    let rf_half = RF / 2;

    // First half full rounds
    for r in 0..rf_half {
        // Add round constants (ARK)
        for i in 0..T {
            state[i] += params.rc_full[r][i];
        }
        // Full S-box layer
        for i in 0..T {
            state[i] = sbox5(state[i]);
        }
        // MDS linear layer
        *state = mds_mul(&params.mds, state);
    }

    // Partial rounds
    for r in 0..RP {
        // ARK on first element
        state[0] += params.rc_partial[r];
        // S-box on first element
        state[0] = sbox5(state[0]);
        // MDS
        *state = mds_mul(&params.mds, state);
    }

    // Second half full rounds
    for r in rf_half..RF {
        for i in 0..T {
            state[i] += params.rc_full[r][i];
        }
        for i in 0..T {
            state[i] = sbox5(state[i]);
        }
        *state = mds_mul(&params.mds, state);
    }
}

// Multiply state vector by MDS matrix: out = M * state
fn mds_mul(mds: &[[F; T]; T], state: &[F; T]) -> [F; T] {
    let mut out = [F::zero(); T];
    for i in 0..T {
        let mut acc = F::zero();
        for j in 0..T {
            acc += mds[i][j] * state[j];
        }
        out[i] = acc;
    }
    out
}

// Public hashing API: absorb chunks with domain-separation tag in capacity slot.
// Returns the first state element as digest.
pub fn hash_with_ds(inputs: &[F], ds_tag: F, params: &PoseidonParams) -> F {
    let mut state = [F::zero(); T];
    // capacity slot set to ds_tag (last element)
    state[T - 1] = ds_tag;

    // absorb RATE elements per block
    for chunk in inputs.chunks(RATE) {
        for (i, &x) in chunk.iter().enumerate() {
            state[i] += x;
        }
        // remaining unused rate lanes unchanged
        permute(&mut state, params);
    }

    state[0]
}

// Parameter generation using utils::fr_from_hash for reproducible constants.
pub mod params {
    use super::*;

    pub fn generate_params_t17_x5(seed: &[u8]) -> PoseidonParams {
        // Deterministically derive MDS and round constants from a seed.
        // Very simple construction for scaffolding/testing; replace with spec-conformant generation if needed.
        let mut mds = [[F::zero(); T]; T];
        for i in 0..T {
            for j in 0..T {
                // Tag: "POSEIDON-MDS", index bytes: i,j, seed
                let tag = "POSEIDON-MDS";
                let mut data = Vec::with_capacity(seed.len() + 16);
                data.extend_from_slice(&(i as u64).to_le_bytes());
                data.extend_from_slice(&(j as u64).to_le_bytes());
                data.extend_from_slice(seed);
                mds[i][j] = fr_from_hash(tag, &data);
            }
        }

        let mut rc_full = [[F::zero(); T]; RF];
        for r in 0..RF {
            for i in 0..T {
                let tag = "POSEIDON-RC-FULL";
                let mut data = Vec::with_capacity(seed.len() + 16);
                data.extend_from_slice(&(r as u64).to_le_bytes());
                data.extend_from_slice(&(i as u64).to_le_bytes());
                data.extend_from_slice(seed);
                rc_full[r][i] = fr_from_hash(tag, &data);
            }
        }

        let mut rc_partial = [F::zero(); RP];
        for r in 0..RP {
            let tag = "POSEIDON-RC-PART";
            let mut data = Vec::with_capacity(seed.len() + 8);
            data.extend_from_slice(&(r as u64).to_le_bytes());
            data.extend_from_slice(seed);
            rc_partial[r] = fr_from_hash(tag, &data);
        }

        PoseidonParams { mds, rc_full, rc_partial }
    }
}
