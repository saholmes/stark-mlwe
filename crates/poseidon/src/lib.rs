use ark_ff::{Field, Zero};
use ark_pallas::Fr as F;
use utils::fr_from_hash;

// Poseidon permutation parameters for benchmarking and M1 scaffolding.
// Width t=17 matches Merkle arity m=16 with capacity c=1.
pub const T: usize = 17;       // state width = rate(16) + capacity(1)
pub const RATE: usize = 16;
pub const CAPACITY: usize = 1;
pub const RF: usize = 8;       // number of full rounds (total)
pub const RP: usize = 64;      // number of partial rounds (t=17)
// For t=9 use RP_9 = 60.
pub const RP_9: usize = 60;
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
        *state = mds_mul_fixed(&params.mds, state);
    }

    // Partial rounds
    for r in 0..RP {
        // ARK on first element
        state[0] += params.rc_partial[r];
        // S-box on first element
        state[0] = sbox5(state[0]);
        // MDS
        *state = mds_mul_fixed(&params.mds, state);
    }

    // Second half full rounds
    for r in rf_half..RF {
        for i in 0..T {
            state[i] += params.rc_full[r][i];
        }
        for i in 0..T {
            state[i] = sbox5(state[i]);
        }
        *state = mds_mul_fixed(&params.mds, state);
    }
}

// Multiply state vector by MDS matrix: out = M * state (fixed T)
fn mds_mul_fixed(mds: &[[F; T]; T], state: &[F; T]) -> [F; T] {
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

// Public hashing API (fixed width): absorb chunks with DS tag in capacity slot.
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

// ========= Milestone 1 additions: dynamic width support and params builder =========

#[derive(Clone, Debug)]
pub struct PoseidonParamsDynamic {
    pub t: usize,                 // state width
    pub rate: usize,              // rate = t - 1 (capacity = 1)
    pub rounds_full: usize,       // RF = 8
    pub rounds_partial: usize,    // RP = 64 (t=17) or 60 (t=9)
    pub alpha: u64,               // 5
    pub mds: Vec<Vec<F>>,         // t x t
    pub rc_full: Vec<Vec<F>>,     // RF x t
    pub rc_partial: Vec<F>,       // RP elements
}

/// Build Poseidon parameters for width t with alpha=5, RF=8, RP in {64,60}.
/// Supported widths: t = 17 (m=16), t = 9 (m=8).
/// Uses deterministic fr_from_hash-based derivation for stability.
/// Swap in audited constants when ready without changing the signature.
pub fn poseidon_params_for_width(t: usize) -> PoseidonParamsDynamic {
    let (rf, rp) = match t {
        17 => (8usize, 64usize),
        9 => (8usize, 60usize),
        _ => panic!("unsupported Poseidon width t={t}; supported t ∈ {{17, 9}}"),
    };
    let rate = t - 1;
    let seed = seed_for_t(t);

    let mds = derive_mds(&seed, t);
    let rc_full = derive_rc_full(&seed, rf, t);
    let rc_partial = derive_rc_partial(&seed, rp);

    PoseidonParamsDynamic {
        t,
        rate,
        rounds_full: rf,
        rounds_partial: rp,
        alpha: 5,
        mds,
        rc_full,
        rc_partial,
    }
}

fn seed_for_t(t: usize) -> Vec<u8> {
    // Distinct seeds per width to avoid accidental collisions.
    let mut s = Vec::new();
    s.extend_from_slice(b"POSEIDON-PALLAS-T");
    s.extend_from_slice(&(t as u64).to_le_bytes());
    s
}

fn derive_mds(seed: &[u8], t: usize) -> Vec<Vec<F>> {
    let mut m = vec![vec![F::zero(); t]; t];
    for i in 0..t {
        for j in 0..t {
            let tag = "POSEIDON-MDS";
            let mut data = Vec::with_capacity(seed.len() + 16);
            data.extend_from_slice(&(i as u64).to_le_bytes());
            data.extend_from_slice(&(j as u64).to_le_bytes());
            data.extend_from_slice(seed);
            m[i][j] = fr_from_hash(tag, &data);
        }
    }
    m
}

fn derive_rc_full(seed: &[u8], rf: usize, t: usize) -> Vec<Vec<F>> {
    let mut rc = vec![vec![F::zero(); t]; rf];
    for r in 0..rf {
        for i in 0..t {
            let tag = "POSEIDON-RC-FULL";
            let mut data = Vec::with_capacity(seed.len() + 16);
            data.extend_from_slice(&(r as u64).to_le_bytes());
            data.extend_from_slice(&(i as u64).to_le_bytes());
            data.extend_from_slice(seed);
            rc[r][i] = fr_from_hash(tag, &data);
        }
    }
    rc
}

fn derive_rc_partial(seed: &[u8], rp: usize) -> Vec<F> {
    let mut rc = vec![F::zero(); rp];
    for r in 0..rp {
        let tag = "POSEIDON-RC-PART";
        let mut data = Vec::with_capacity(seed.len() + 8);
        data.extend_from_slice(&(r as u64).to_le_bytes());
        data.extend_from_slice(seed);
        rc[r] = fr_from_hash(tag, &data);
    }
    rc
}

/// Generic permutation for dynamic params (t ∈ {9, 17}).
pub fn permute_dynamic(state: &mut [F], params: &PoseidonParamsDynamic) {
    let t = params.t;
    assert_eq!(state.len(), t);

    let rf = params.rounds_full;
    let rp = params.rounds_partial;
    let rf_half = rf / 2;

    // First half full rounds
    for r in 0..rf_half {
        // ARK
        for i in 0..t {
            state[i] += params.rc_full[r][i];
        }
        // Full S-box
        for i in 0..t {
            state[i] = sbox5(state[i]);
        }
        // MDS
        mds_mul_dynamic_in_place(&params.mds, state);
    }

    // Partial rounds (S-box on lane 0)
    for r in 0..rp {
        state[0] += params.rc_partial[r];
        state[0] = sbox5(state[0]);
        mds_mul_dynamic_in_place(&params.mds, state);
    }

    // Second half full rounds
    for r in rf_half..rf {
        for i in 0..t {
            state[i] += params.rc_full[r][i];
        }
        for i in 0..t {
            state[i] = sbox5(state[i]);
        }
        mds_mul_dynamic_in_place(&params.mds, state);
    }
}

fn mds_mul_dynamic_in_place(mds: &[Vec<F>], state: &mut [F]) {
    let t = state.len();
    debug_assert_eq!(mds.len(), t);
    let mut out = vec![F::zero(); t];
    for i in 0..t {
        let mut acc = F::zero();
        for j in 0..t {
            acc += mds[i][j] * state[j];
        }
        out[i] = acc;
    }
    state.copy_from_slice(&out);
}

/// Absorb one element into the sponge rate; permute when the rate is full.
#[inline]
fn absorb_one(x: F, state: &mut [F], cursor: &mut usize, rate: usize, params: &PoseidonParamsDynamic) {
    state[*cursor] += x;
    *cursor += 1;
    if *cursor == rate {
        *cursor = 0;
        permute_dynamic(state, params);
    }
}

/// DS-friendly hash for dynamic widths (rate = t-1, capacity=1).
/// Absorbs ds_fields first, then inputs (children) in order, padding with 1 then 0s.
/// Returns state[0] as the digest.
pub fn hash_with_ds_dynamic(ds_fields: &[F], inputs: &[F], params: &PoseidonParamsDynamic) -> F {
    let t = params.t;
    let rate = params.rate;
    assert_eq!(rate + 1, t);

    let mut state = vec![F::zero(); t];
    let mut cursor = 0usize;

    // Absorb DS preamble
    for &x in ds_fields {
        absorb_one(x, &mut state, &mut cursor, rate, params);
    }
    // Absorb message/children
    for &x in inputs {
        absorb_one(x, &mut state, &mut cursor, rate, params);
    }
    // Padding: 1 then zeros until block boundary
    absorb_one(F::from(1u64), &mut state, &mut cursor, rate, params);
    while cursor != 0 {
        absorb_one(F::zero(), &mut state, &mut cursor, rate, params);
    }

    // Squeeze first element
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

// ======================= NEW: Static <-> Dynamic adapters =======================

impl From<&PoseidonParams> for PoseidonParamsDynamic {
    fn from(p: &PoseidonParams) -> Self {
        let t = T;
        let rate = RATE;
        let rf = RF;
        let rp = RP;

        // MDS to Vec<Vec<F>>
        let mut mds_v = vec![vec![F::zero(); t]; t];
        for i in 0..t {
            for j in 0..t {
                mds_v[i][j] = p.mds[i][j];
            }
        }

        // rc_full RF x t
        let mut rc_full_v = vec![vec![F::zero(); t]; rf];
        for r in 0..rf {
            for i in 0..t {
                rc_full_v[r][i] = p.rc_full[r][i];
            }
        }

        // rc_partial RP
        let mut rc_partial_v = vec![F::zero(); rp];
        for r in 0..rp {
            rc_partial_v[r] = p.rc_partial[r];
        }

        PoseidonParamsDynamic {
            t,
            rate,
            rounds_full: rf,
            rounds_partial: rp,
            alpha: ALPHA,
            mds: mds_v,
            rc_full: rc_full_v,
            rc_partial: rc_partial_v,
        }
    }
}

/// Convenience wrapper for t=17 static -> dynamic.
pub fn dynamic_from_static_t17(p: &PoseidonParams) -> PoseidonParamsDynamic {
    PoseidonParamsDynamic::from(p)
}

/// Optional helper: only valid when params describe t=17.
pub fn static_from_dynamic_t17(d: &PoseidonParamsDynamic) -> PoseidonParams {
    assert_eq!(d.t, T, "static_from_dynamic_t17 requires t=17");
    assert_eq!(d.rounds_full, RF, "rounds_full mismatch");
    assert_eq!(d.rounds_partial, RP, "rounds_partial mismatch (expected RP={})", RP);

    let mut mds = [[F::zero(); T]; T];
    for i in 0..T {
        for j in 0..T {
            mds[i][j] = d.mds[i][j];
        }
    }
    let mut rc_full = [[F::zero(); T]; RF];
    for r in 0..RF {
        for i in 0..T {
            rc_full[r][i] = d.rc_full[r][i];
        }
    }
    let mut rc_partial = [F::zero(); RP];
    for r in 0..RP {
        rc_partial[r] = d.rc_partial[r];
    }
    PoseidonParams { mds, rc_full, rc_partial }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::params::generate_params_t17_x5;

    #[test]
    fn static_dynamic_roundtrip_t17() {
        let seed = b"POSEIDON-T17-X5-SEED";
        let p_static = generate_params_t17_x5(seed);
        let d = PoseidonParamsDynamic::from(&p_static);
        assert_eq!(d.t, T);
        assert_eq!(d.rate, RATE);
        assert_eq!(d.rounds_full, RF);
        assert_eq!(d.rounds_partial, RP);
        assert_eq!(d.alpha, ALPHA);

        let p_back = static_from_dynamic_t17(&d);
        // Compare a few entries to ensure mapping is correct
        assert_eq!(p_back.mds[0][0], p_static.mds[0][0]);
        assert_eq!(p_back.rc_full[0][0], p_static.rc_full[0][0]);
        assert_eq!(p_back.rc_partial[0], p_static.rc_partial[0]);
    }
}