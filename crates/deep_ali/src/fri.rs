//! DEEP-ALI + DEEP-FRI with DS-aware high-arity Merkle (merkle crate) and bucket multiproofs.
//! FS challenges are derived via the transcript crate (Poseidon over ark 0.5, width t=17).

use ark_ff::{Field, One, Zero};
use ark_pallas::Fr as F;
use ark_serialize::CanonicalSerialize;
use rand::{rngs::StdRng, Rng, SeedableRng};

use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
use ark_poly::EvaluationDomain;

// DS-aware Merkle from your merkle crate
use merkle::{MerkleChannelCfg, MerkleProof, MerkleProver, MerkleTree};

// Transcript-based FS (Poseidon-based RO in your repo)
use transcript::{default_params as transcript_params, Transcript};

/* ============================== Domain separation tags ============================== */

mod ds {
    pub const FRI_SEED: &[u8] = b"FRI/seed";
    pub const FRI_INDEX: &[u8] = b"FRI/index";
    pub const FRI_Z_L: &[u8] = b"FRI/z/l";
}

/* ============================ Transcript helpers ============================ */

fn tr_hash_fields_tagged(tag: &[u8], fields: &[F]) -> F {
    let mut tr = Transcript::new(b"FRI/FS", transcript_params());
    tr.absorb_bytes(tag);
    for &x in fields {
        tr.absorb_field(x);
    }
    tr.challenge(b"out")
}

/* ============================ FRI domain structures ============================ */

#[derive(Clone, Copy, Debug)]
pub struct FriDomain {
    pub omega: F,
    pub size: usize,
}

impl FriDomain {
    pub fn new_radix2(size: usize) -> Self {
        let dom = Domain::<F>::new(size).expect("radix-2 domain exists");
        Self {
            omega: dom.group_gen,
            size,
        }
    }
}

/* ============================== FS challenges ============================== */

pub fn fri_sample_z_ell(seed_z: u64, level: usize, domain_size: usize) -> F {
    // Sample from transcript-derived RNG seed; reject if in H (z^N = 1).
    let fused = tr_hash_fields_tagged(
        ds::FRI_Z_L,
        &[
            F::from(seed_z),
            F::from(level as u64),
            F::from(domain_size as u64),
        ],
    );
    let mut seed_bytes = [0u8; 32];
    fused
        .serialize_uncompressed(&mut seed_bytes[..])
        .expect("serialize");
    let mut rng = StdRng::from_seed(seed_bytes);
    loop {
        let cand = F::from(rng.gen::<u64>());
        if cand.pow(&[domain_size as u64, 0, 0, 0]) != F::one() {
            return cand;
        }
    }
}

/* ================================ Folding ================================= */

pub fn fri_fold_layer(f_l: &[F], z_l: F, m: usize) -> Vec<F> {
    assert!(m >= 2);
    assert!(f_l.len() % m == 0, "layer size must be divisible by m");
    let n_next = f_l.len() / m;
    let mut out = vec![F::zero(); n_next];

    // Precompute z powers
    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m {
        z_pows.push(acc);
        acc *= z_l;
    }

    for b in 0..n_next {
        let base = b * m;
        let mut s = F::zero();
        for t in 0..m {
            s += f_l[base + t] * z_pows[t];
        }
        out[b] = s;
    }
    out
}

pub fn fri_fold_schedule(f0: Vec<F>, schedule: &[usize], seed: u64) -> Vec<Vec<F>> {
    let mut layers = Vec::with_capacity(schedule.len() + 1);
    let mut cur = f0;
    let mut cur_size = cur.len();
    layers.push(cur.clone());

    for (level, &m) in schedule.iter().enumerate() {
        assert!(
            cur_size % m == 0,
            "size must be divisible by m at level {level}"
        );
        let z_l = fri_sample_z_ell(seed, level, cur_size);
        cur = fri_fold_layer(&cur, z_l, m);
        cur_size = cur.len();
        layers.push(cur.clone());
    }
    layers
}

/* ============================== Combined leaf type ============================== */

#[derive(Clone, Copy, Debug)]
pub struct CombinedLeaf {
    pub f: F,
    pub cp: F,
}

/* ============================= CP computation ============================= */

pub fn compute_cp_layer(f_l: &[F], f_l_plus_1: &[F], z_l: F, m: usize, omega_l: F) -> Vec<F> {
    assert!(f_l.len() % m == 0, "size must be divisible by m");
    let n = f_l.len();
    let n_next = n / m;
    assert_eq!(
        f_l_plus_1.len(),
        n_next,
        "f_l_plus_1 length mismatch at cp layer"
    );

    // Precompute z^t (t=0..m-1)
    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m {
        z_pows.push(acc);
        acc *= z_l;
    }

    // Precompute omega^i (i=0..n-1), natural order
    let mut omega_pows = Vec::with_capacity(n);
    let mut w = F::one();
    for _ in 0..n {
        omega_pows.push(w);
        w *= omega_l;
    }

    // Residual per bucket R_b = sum_t f[bm+t] z^t - f_{l+1}[b]
    let mut residuals = vec![F::zero(); n_next];
    for b in 0..n_next {
        let base = b * m;
        let mut s = F::zero();
        for t in 0..m {
            s += f_l[base + t] * z_pows[t];
        }
        residuals[b] = s - f_l_plus_1[b];
    }

    // cp[i] = R_{i/m} / (z - omega^i)
    let mut cp = vec![F::zero(); n];
    for i in 0..n {
        let b = i / m;
        let denom = z_l - omega_pows[i];
        let inv = denom.inverse().expect("z not in H_l");
        cp[i] = residuals[b] * inv;
    }
    cp
}

pub fn build_combined_layer(f_l: &[F], cp_l: &[F]) -> Vec<CombinedLeaf> {
    assert_eq!(f_l.len(), cp_l.len());
    f_l.iter()
        .zip(cp_l.iter())
        .map(|(&f, &cp)| CombinedLeaf { f, cp })
        .collect()
}

/* ============================= Helpers: sizes and domains ============================= */

fn layer_sizes_from_schedule(n0: usize, schedule: &[usize]) -> Vec<usize> {
    let mut sizes = Vec::with_capacity(schedule.len() + 1);
    let mut n = n0;
    sizes.push(n);
    for &m in schedule {
        assert!(n % m == 0, "schedule not dividing domain size");
        n /= m;
        sizes.push(n);
    }
    sizes
}

// Exact domain generator per layer size N_ell
fn layer_domains_from_schedule(n0: usize, schedule: &[usize]) -> Vec<(usize, F)> {
    let sizes = layer_sizes_from_schedule(n0, schedule);
    let mut out = Vec::with_capacity(schedule.len());
    for ell in 0..schedule.len() {
        let n = sizes[ell];
        let dom = Domain::<F>::new(n).expect("radix-2");
        out.push((n, dom.group_gen));
    }
    out
}

/* ============================= Local check verify ============================= */

fn verify_local_check_bucket_sum(
    bucket_f: &[F], // length <= m (canonical order within bucket)
    child_leaf_i: CombinedLeaf,
    parent_leaf_b: CombinedLeaf,
    z_l: F,
    m: usize,
    omega_l: F,
    i: usize,
    n_layer: usize,
) -> bool {
    // Reconstruct RHS: Σ_t bucket_f[t] z^t (pad with zeros to length m)
    let mut rhs = F::zero();
    let mut pow = F::one();
    for t in 0..m {
        let v = if t < bucket_f.len() { bucket_f[t] } else { F::zero() };
        rhs += v * pow;
        pow *= z_l;
    }

    // Compute ω^i with i reduced modulo layer size
    let w_i = omega_l.pow(&[(i % n_layer) as u64, 0, 0, 0]);

    // LHS = cp(i) * (z - w_i) + f_{l+1}(b)
    let lhs = child_leaf_i.cp * (z_l - w_i) + parent_leaf_b.f;

    lhs == rhs
}

/* ============================= FS helpers ============================= */

fn fs_seed_from_roots(roots: &[F]) -> F {
    tr_hash_fields_tagged(ds::FRI_SEED, roots)
}

fn index_from_seed(seed_f: F, n_pow2: usize) -> usize {
    assert!(n_pow2.is_power_of_two());
    let mask = n_pow2 - 1;
    let mut seed_bytes = [0u8; 32];
    seed_f.serialize_uncompressed(&mut seed_bytes[..]).unwrap();
    let mut rng = StdRng::from_seed(seed_bytes);
    (rng.gen::<u64>() as usize) & mask
}

fn index_seed(roots_seed: F, ell: usize, q: usize) -> F {
    tr_hash_fields_tagged(
        ds::FRI_INDEX,
        &[roots_seed, F::from(ell as u64), F::from(q as u64)],
    )
}

/* ============================= Transcript structures ============================= */

#[derive(Clone)]
pub struct FriLayerCommitment {
    pub n: usize,
    pub m: usize,
    pub root: F,
    pub f: Vec<F>,
    pub cp: Vec<F>,
    pub tree: MerkleTree,
    pub cfg: MerkleChannelCfg,
}

#[derive(Clone)]
pub struct FriTranscript {
    pub schedule: Vec<usize>,
    pub layers: Vec<FriLayerCommitment>,
}

pub struct FriProverParams {
    pub schedule: Vec<usize>,
    pub seed_z: u64,
}

pub struct FriProverState {
    pub f_layers: Vec<Vec<F>>,
    pub cp_layers: Vec<Vec<F>>,
    pub transcript: FriTranscript,
    pub omega_layers: Vec<F>,
    pub z_layers: Vec<F>,
}

/* ============================= Build transcript ============================= */

pub fn fri_build_transcript(
    f0: Vec<F>,
    domain0: FriDomain,
    params: &FriProverParams,
) -> FriProverState {
    let schedule = params.schedule.clone();
    let l = schedule.len();

    let layer_domains = layer_domains_from_schedule(domain0.size, &schedule);

    let mut f_layers = Vec::with_capacity(l + 1);
    let mut z_layers = Vec::with_capacity(l);
    let mut omega_layers = Vec::with_capacity(l);
    let mut cur_f = f0;
    let mut cur_size = domain0.size;
    f_layers.push(cur_f.clone());

    for (ell, &m) in schedule.iter().enumerate() {
        let z = fri_sample_z_ell(params.seed_z, ell, cur_size);
        z_layers.push(z);
        let (_n_ell, omega_ell) = layer_domains[ell];
        omega_layers.push(omega_ell);
        cur_f = fri_fold_layer(&cur_f, z, m);
        cur_size /= m;
        f_layers.push(cur_f.clone());
    }

    let mut cp_layers = Vec::with_capacity(l + 1);
    for ell in 0..l {
        let m = schedule[ell];
        let z = z_layers[ell];
        let omega = omega_layers[ell];
        let cp = compute_cp_layer(&f_layers[ell], &f_layers[ell + 1], z, m, omega);
        cp_layers.push(cp);
    }
    cp_layers.push(vec![F::zero(); f_layers[l].len()]);

    // Commit each layer as a combined-leaf Merkle via merkle::MerkleProver
    let mut layers = Vec::with_capacity(l + 1);
    for ell in 0..=l {
        let n = f_layers[ell].len();
        let m_ell = if ell < l { schedule[ell] } else { 1 };
        let arity = if m_ell >= 2 { m_ell } else { 1 };
        let cfg = MerkleChannelCfg::new(arity).with_tree_label(ell as u64);

        // For the paper’s preset you typically see m in {16,16,8,1}; assert lightly in debug.
        debug_assert!(
            arity == 1 || arity == 2 || arity == 8 || arity == 16,
            "unexpected arity {} at layer {}",
            arity,
            ell
        );

        let prover = MerkleProver::new(cfg.clone());
        let (root, tree) = prover.commit_pairs(&f_layers[ell][..], &cp_layers[ell][..]);

        layers.push(FriLayerCommitment {
            n,
            m: m_ell,
            root,
            f: f_layers[ell].clone(),
            cp: cp_layers[ell].clone(),
            tree,
            cfg,
        });
    }

    FriProverState {
        f_layers,
        cp_layers,
        transcript: FriTranscript { schedule, layers },
        omega_layers,
        z_layers,
    }
}

/* ============================= Query openings (multi-proof) ============================= */

#[derive(Clone)]
pub struct LayerOpenings {
    pub i: usize,
    pub child_indices: Vec<usize>, // full bucket indices (length up to m)
    pub child_pairs: Vec<(F, F)>,  // (f, cp) for bucket (ordered)
    pub child_proof: MerkleProof,  // multiproof for the bucket
    pub parent_index: usize,       // b = i / m
    pub parent_pair: (F, F),       // (f_{l+1}[b], cp_{l+1}[b]) (cp often zero at last layer)
    pub parent_proof: MerkleProof, // single-index multiproof
}

#[derive(Clone)]
pub struct FriQueryOpenings {
    pub per_layer: Vec<LayerOpenings>,
    pub final_index: usize, // usually 0 if last layer size is 1
    pub final_pair: (F, F),
    pub final_proof: MerkleProof,
}

pub fn fri_prove_queries(
    st: &FriProverState,
    r: usize,
    roots_seed: F,
) -> (Vec<FriQueryOpenings>, Vec<F>) {
    let l = st.transcript.schedule.len();
    let mut all = Vec::with_capacity(r);

    for q in 0..r {
        let mut per_layer = Vec::with_capacity(l);
        for ell in 0..l {
            let layer = &st.transcript.layers[ell];
            let n = layer.n;
            let n_pow2 = n.next_power_of_two();
            let m = layer.m;

            let seed = index_seed(roots_seed, ell, q);
            let i0 = index_from_seed(seed, n_pow2);
            let i = if i0 < n {
                i0
            } else {
                // two-stage reseed if overshoot due to next_power_of_two masking
                let reseed = tr_hash_fields_tagged(ds::FRI_INDEX, &[seed, F::from(1u64)]);
                let i2 = index_from_seed(reseed, n_pow2);
                if i2 < n { i2 } else { i2 & (n - 1) }
            };

            // derive parent bucket
            let b = i / m;
            let base = b * m;
            let end = core::cmp::min(base + m, n);
            let bucket_len = end - base;

            // Prepare bucket indices and pairs
            let child_indices: Vec<usize> = (0..bucket_len).map(|t| base + t).collect();
            let child_pairs: Vec<(F, F)> =
                child_indices.iter().map(|&j| (layer.f[j], layer.cp[j])).collect();

            // Multiproof for full bucket at child layer
            let child_proof = layer.tree.open_many(&child_indices);

            // Parent index and pair
            let parent_layer = &st.transcript.layers[ell + 1];
            let parent_index = b;
            let parent_pair = (parent_layer.f[parent_index], parent_layer.cp[parent_index]);
            let parent_proof = parent_layer.tree.open_many(&[parent_index]);

            per_layer.push(LayerOpenings {
                i,
                child_indices,
                child_pairs,
                child_proof,
                parent_index,
                parent_pair,
                parent_proof,
            });
        }
        let last = &st.transcript.layers[l];
        let final_index = 0usize;
        let final_pair = (last.f[final_index], last.cp[final_index]);
        let final_proof = last.tree.open_many(&[final_index]);
        all.push(FriQueryOpenings {
            per_layer,
            final_index,
            final_pair,
            final_proof,
        });
    }

    let roots: Vec<F> = st.transcript.layers.iter().map(|l| l.root).collect();
    (all, roots)
}

/* ================================ Verifier ================================= */

pub fn fri_verify_queries(
    schedule: &[usize],
    n0: usize,
    _omega0_unused: F,
    seed_z: u64,
    roots: &[F],
    queries: &[FriQueryOpenings],
    r: usize,
) -> bool {
    let l = schedule.len();
    if roots.len() != l + 1 {
        return false;
    }

    let sizes = layer_sizes_from_schedule(n0, schedule);
    let layer_domains = layer_domains_from_schedule(n0, schedule);

    // z_ell from FS seed
    let mut z_layers = Vec::with_capacity(l);
    for ell in 0..l {
        z_layers.push(fri_sample_z_ell(seed_z, ell, sizes[ell]));
    }

    let roots_seed = fs_seed_from_roots(roots);

    for (q, qopen) in queries.iter().enumerate().take(r) {
        if qopen.per_layer.len() != l {
            return false;
        }

        for ell in 0..l {
            let n = sizes[ell];
            let n_pow2 = n.next_power_of_two();
            let m = schedule[ell];

            // Recompute derived i
            let seed = index_seed(roots_seed, ell, q);
            let i0 = index_from_seed(seed, n_pow2);
            let derived_i = if i0 < n {
                i0
            } else {
                let reseed = tr_hash_fields_tagged(ds::FRI_INDEX, &[seed, F::from(1u64)]);
                let i2 = index_from_seed(reseed, n_pow2);
                if i2 < n { i2 } else { i2 & (n - 1) }
            };

            let lay = &qopen.per_layer[ell];
            if lay.i != derived_i {
                return false;
            }

            // Verify child bucket multiproof against child root
            let child_root = roots[ell];
            let prover_child =
                MerkleProver::new(MerkleChannelCfg::new(m).with_tree_label(ell as u64));
            if lay.child_indices.is_empty() || lay.child_indices.len() > m {
                return false;
            }
            if !prover_child.verify_pairs(
                &child_root,
                &lay.child_indices[..],
                &lay.child_pairs[..],
                &lay.child_proof,
            ) {
                return false;
            }

            // Parent verification
            let m_parent = if ell + 1 < l { schedule[ell + 1] } else { 1 };
            let parent_root = roots[ell + 1];
            let prover_parent = MerkleProver::new(
                MerkleChannelCfg::new(m_parent).with_tree_label((ell + 1) as u64),
            );
            if !prover_parent.verify_pairs(
                &parent_root,
                core::slice::from_ref(&lay.parent_index),
                core::slice::from_ref(&lay.parent_pair),
                &lay.parent_proof,
            ) {
                return false;
            }

            // Local check using bucket f-values
            let (n_layer, omega_l) = layer_domains[ell];
            let z = z_layers[ell];

            // Build bucket_f in canonical t-order: child_indices are base..base+bucket_len increasing
            let bucket_f: Vec<F> = lay.child_pairs.iter().map(|p| p.0).collect();

            let child_index = lay.i;
            let b = child_index / m;
            let t_pos = child_index - (b * m);
            if t_pos >= m {
                return false;
            }
            let child_leaf_i = {
                // Find (f,cp) for i inside child_pairs at pos t_pos
                if t_pos >= lay.child_pairs.len() {
                    // If bucket tail was shorter than m, we cannot have queried such i.
                    return false;
                }
                let (f_i, cp_i) = lay.child_pairs[t_pos];
                CombinedLeaf { f: f_i, cp: cp_i }
            };

            let parent_leaf_b = CombinedLeaf {
                f: lay.parent_pair.0,
                cp: lay.parent_pair.1,
            };

            if !verify_local_check_bucket_sum(
                &bucket_f,
                child_leaf_i,
                parent_leaf_b,
                z,
                m,
                omega_l,
                child_index,
                n_layer,
            ) {
                return false;
            }
        }

        // Final layer leaf check
        let last_root = roots[l];
        let prover_last = MerkleProver::new(MerkleChannelCfg::new(1).with_tree_label(l as u64));
        if !prover_last.verify_pairs(
            &last_root,
            core::slice::from_ref(&qopen.final_index),
            core::slice::from_ref(&qopen.final_pair),
            &qopen.final_proof,
        ) {
            return false;
        }
    }

    true
}

/* ============================= Orchestrators ============================= */

pub type AliA = Vec<F>;
pub type AliS = Vec<F>;
pub type AliE = Vec<F>;
pub type AliT = Vec<F>;

pub trait DeepAliBuilder {
    fn build_f0(
        &self,
        a: &AliA,
        s: &AliS,
        e: &AliE,
        t: &AliT,
        n0: usize,
        domain: FriDomain,
    ) -> Vec<F>;
}

#[derive(Clone, Default)]
pub struct DeepAliMock;

fn tr_hash_many(tag: &[u8], xs: &[F]) -> F {
    tr_hash_fields_tagged(tag, xs)
}

impl DeepAliBuilder for DeepAliMock {
    fn build_f0(
        &self,
        a: &AliA,
        s: &AliS,
        e: &AliE,
        t: &AliT,
        n0: usize,
        _domain: FriDomain,
    ) -> Vec<F> {
        let seed_f = tr_hash_fields_tagged(
            b"ALI/mock/seed",
            &[
                tr_hash_many(b"ALI/a", a),
                tr_hash_many(b"ALI/s", s),
                tr_hash_many(b"ALI/e", e),
                tr_hash_many(b"ALI/t", t),
                F::from(n0 as u64),
            ],
        );
        let mut seed_bytes = [0u8; 32];
        seed_f.serialize_uncompressed(&mut seed_bytes[..]).unwrap();
        let mut rng = StdRng::from_seed(seed_bytes);
        (0..n0).map(|_| F::from(rng.gen::<u64>())).collect()
    }
}

#[derive(Clone)]
pub struct DeepFriParams {
    pub schedule: Vec<usize>,
    pub r: usize,
    pub seed_z: u64,
}

pub struct DeepFriProof {
    pub roots: Vec<F>,
    pub queries: Vec<FriQueryOpenings>,
    pub n0: usize,
    pub omega0: F,
}

pub fn deep_fri_prove<B: DeepAliBuilder>(
    builder: &B,
    a: &AliA,
    s: &AliS,
    e: &AliE,
    t: &AliT,
    n0: usize,
    params: &DeepFriParams,
) -> DeepFriProof {
    let domain0 = FriDomain::new_radix2(n0);
    let f0 = builder.build_f0(a, s, e, t, n0, domain0);

    let st = fri_build_transcript(
        f0,
        domain0,
        &FriProverParams {
            schedule: params.schedule.clone(),
            seed_z: params.seed_z,
        },
    );

    let roots: Vec<F> = st.transcript.layers.iter().map(|l| l.root).collect();
    let roots_seed = fs_seed_from_roots(&roots);
    let (queries, roots2) = fri_prove_queries(&st, params.r, roots_seed);
    debug_assert_eq!(roots, roots2);

    DeepFriProof {
        roots,
        queries,
        n0,
        omega0: domain0.omega,
    }
}

pub fn deep_fri_verify(params: &DeepFriParams, proof: &DeepFriProof) -> bool {
    fri_verify_queries(
        &params.schedule,
        proof.n0,
        proof.omega0,
        params.seed_z,
        &proof.roots,
        &proof.queries,
        params.r,
    )
}

/* ===================================== Tests / Bench preset ===================================== */

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;

    #[test]
    fn test_fri_fold_shape() {
        let n0 = 4096usize;
        let schedule = [16usize, 16usize, 8usize];
        let mut rng = StdRng::seed_from_u64(7777);
        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();
        let layers = fri_fold_schedule(f0.clone(), &schedule, 0xDEADBEEF);
        assert_eq!(layers.len(), schedule.len() + 1);
        assert_eq!(layers[0].len(), n0);
        assert_eq!(layers[1].len(), n0 / 16);
        assert_eq!(layers[2].len(), n0 / 16 / 16);
        assert_eq!(layers[3].len(), n0 / 16 / 16 / 8);
    }

    #[test]
    fn test_combined_layer_merkle_roundtrip_and_local_verify_multiproof() {
        let n0 = 1024usize;
        let m = 16usize;
        let dom = FriDomain::new_radix2(n0);
        let omega0 = dom.omega;

        let mut rng = StdRng::seed_from_u64(1212);
        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();
        let z0 = fri_sample_z_ell(0xDEAD_BEEF, 0, n0);
        let f1 = fri_fold_layer(&f0, z0, m);

        let cp0 = compute_cp_layer(&f0, &f1, z0, m, omega0);

        // Commit layers using DS-aware Merkle
        let cfg0 = MerkleChannelCfg::new(16).with_tree_label(0);
        let prover0 = MerkleProver::new(cfg0.clone());
        let (root0, tree0) = prover0.commit_pairs(&f0[..], &cp0[..]);

        let cp1_dummy = vec![F::zero(); f1.len()];
        let cfg1 = MerkleChannelCfg::new(16).with_tree_label(1);
        let prover1 = MerkleProver::new(cfg1.clone());
        let (root1, tree1) = prover1.commit_pairs(&f1[..], &cp1_dummy[..]);

        // Query random i; open full bucket multiproof and parent
        let mut rngq = StdRng::seed_from_u64(333);
        for _ in 0..10 {
            let i = rngq.gen::<usize>() % n0;
            let b = i / m;
            let base = b * m;
            let end = core::cmp::min(base + m, n0);
            let bucket_len = end - base;

            let child_indices: Vec<usize> = (0..bucket_len).map(|t| base + t).collect();
            let child_pairs: Vec<(F, F)> =
                child_indices.iter().map(|&j| (f0[j], cp0[j])).collect();
            let child_proof = tree0.open_many(&child_indices);

            // Verify child multiproof with facade
            let prov_child = MerkleProver::new(MerkleChannelCfg::new(16).with_tree_label(0));
            assert!(prov_child.verify_pairs(
                &root0,
                &child_indices[..],
                &child_pairs[..],
                &child_proof
            ));

            // Parent single opening
            let parent_index = b;
            let parent_pair = (f1[parent_index], F::zero());
            let parent_proof = tree1.open_many(&[parent_index]);
            let prov_parent = MerkleProver::new(MerkleChannelCfg::new(16).with_tree_label(1));
            assert!(prov_parent.verify_pairs(
                &root1,
                core::slice::from_ref(&parent_index),
                core::slice::from_ref(&parent_pair),
                &parent_proof
            ));

            // Local check using bucket_f
            let bucket_f: Vec<F> = child_pairs.iter().map(|p| p.0).collect();
            let child_leaf_i = CombinedLeaf { f: f0[i], cp: cp0[i] };
            let parent_leaf_b = CombinedLeaf { f: f1[b], cp: F::zero() };
            assert!(super::verify_local_check_bucket_sum(
                &bucket_f, child_leaf_i, parent_leaf_b, z0, m, omega0, i, n0
            ));
        }
    }

    #[test]
    fn test_fri_queries_roundtrip_multiproof() {
        let n0 = 128usize;
        let schedule = vec![8usize, 8usize, 2usize];
        let r = 32usize;

        let dom = FriDomain::new_radix2(n0);
        let mut rng = StdRng::seed_from_u64(2025);
        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();

        let seed_z = 0x1234_5678_ABCDu64;
        let st = fri_build_transcript(
            f0,
            dom,
            &FriProverParams {
                schedule: schedule.clone(),
                seed_z,
            },
        );
        let roots: Vec<F> = st.transcript.layers.iter().map(|l| l.root).collect();
        let roots_seed = super::fs_seed_from_roots(&roots);
        let (queries, roots2) = fri_prove_queries(&st, r, roots_seed);
        assert_eq!(roots, roots2);

        let ok = fri_verify_queries(&schedule, n0, F::one(), seed_z, &roots, &queries, r);
        assert!(ok);
    }

    // “Paper preset” smoke test: N0=2048, schedule (16,16,8), r=32
    #[test]
    fn bench_paper_preset_like() {
        let n0 = 2048usize;
        let schedule = vec![16usize, 16usize, 8usize];
        let r = 32usize;

        let dom = FriDomain::new_radix2(n0);
        let mut rng = StdRng::seed_from_u64(777777);
        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();

        let seed_z = 0xDEEF_BAAD_u64;
        let st = fri_build_transcript(
            f0,
            dom,
            &FriProverParams {
                schedule: schedule.clone(),
                seed_z,
            },
        );

        // Commitments roots
        let roots: Vec<F> = st.transcript.layers.iter().map(|l| l.root).collect();

        // Derive FS query seed from roots and produce r queries
        let roots_seed = super::fs_seed_from_roots(&roots);
        let (queries, roots2) = fri_prove_queries(&st, r, roots_seed);
        assert_eq!(roots, roots2);

        // Verify
        let ok = fri_verify_queries(&schedule, n0, F::one(), seed_z, &roots, &queries, r);
        assert!(ok);

        // Rough “proof size” proxy: count field elements and siblings in proofs
        let mut felts = 0usize;
        for q in &queries {
            for lay in &q.per_layer {
                felts += lay.child_pairs.len() * 2; // (f,cp) per bucket element
                felts += lay
                    .child_proof
                    .siblings
                    .iter()
                    .map(|g| g.len())
                    .sum::<usize>();
                felts += 2; // parent (f,cp)
                felts += lay
                    .parent_proof
                    .siblings
                    .iter()
                    .map(|g| g.len())
                    .sum::<usize>();
            }
            felts += 2; // final layer pair
            felts += q
                .final_proof
                .siblings
                .iter()
                .map(|g| g.len())
                .sum::<usize>();
        }
        eprintln!(
            "approx field elements referenced in proof (not serialized): {}",
            felts
        );
    }
}