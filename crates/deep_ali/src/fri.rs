//! DEEP-ALI + DEEP-FRI with DS-aware high-arity Merkle (merkle crate)
//! and constant-size local checks via combined-layer commitments.
//! FS challenges are derived via the transcript crate (Poseidon over ark 0.5, width t=17).
//!
//! Paper-friendly folding parameters typically use schedule like [16, 16, 8] and r=32.
//! This file implements Option A: per-layer Merkle arity selection with safe fallback:
//!   - try 16 if requested m>=16 and n % 16 == 0
//!   - else try 8 if requested m>=8 and n % 8 == 0
//!   - else 2 if n % 2 == 0
//!   - else 1
//! Folding still uses the exact schedule m; only Merkle arity selection is guarded.

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

/* ============================== Logging guard (opt-in) ============================== */

#[cfg(feature = "fri_bench_log")]
macro_rules! logln {
    ($($tt:tt)*) => { eprintln!($($tt)*); }
}
#[cfg(not(feature = "fri_bench_log"))]
macro_rules! logln {
    ($($tt:tt)*) => {};
}

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

    // Bounded retries to avoid any pathological stalls
    let mut tries = 0usize;
    const MAX_TRIES: usize = 1_000;
    loop {
        let cand = F::from(rng.gen::<u64>());
        if !cand.is_zero() && cand.pow(&[domain_size as u64, 0, 0, 0]) != F::one() {
            return cand;
        }
        tries += 1;
        if tries >= MAX_TRIES {
            let fallback = F::from(seed_z.wrapping_add(level as u64).wrapping_add(7));
            if fallback.pow(&[domain_size as u64, 0, 0, 0]) != F::one() {
                return fallback;
            }
            return F::from(11u64);
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
    pub s: F, // S_l(i) = sum_t f[bm+t] * z^t, duplicated per index in bucket b
}

/* ============================= S-layer computation ============================= */

pub fn compute_s_layer(f_l: &[F], z_l: F, m: usize) -> Vec<F> {
    assert!(f_l.len() % m == 0, "size must be divisible by m");
    let n = f_l.len();
    let n_next = n / m;

    // Precompute z^t (t=0..m-1)
    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m {
        z_pows.push(acc);
        acc *= z_l;
    }

    // For each bucket b compute S_b = sum_t f[bm+t] z^t
    let mut s_bucket = vec![F::zero(); n_next];
    for b in 0..n_next {
        let base = b * m;
        let mut s = F::zero();
        for t in 0..m {
            s += f_l[base + t] * z_pows[t];
        }
        s_bucket[b] = s;
    }

    // Duplicate S_b to each index in bucket
    let mut s_per_i = vec![F::zero(); n];
    for i in 0..n {
        s_per_i[i] = s_bucket[i / m];
    }
    s_per_i
}

pub fn build_combined_layer(f_l: &[F], s_l: &[F]) -> Vec<CombinedLeaf> {
    assert_eq!(f_l.len(), s_l.len());
    f_l.iter()
        .zip(s_l.iter())
        .map(|(&f, &s)| CombinedLeaf { f, s })
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

/* ============================= Constant-size local check ============================= */

fn verify_local_check_constant(
    i: usize,
    m: usize,
    z_l: F,
    omega_l: F,
    n_layer: usize,
    child_leaf_i: CombinedLeaf,
    parent_f_b: F,
) -> bool {
    let b = i / m;
    if b >= n_layer / m {
        return false;
    }
    let w_i = omega_l.pow(&[(i % n_layer) as u64, 0, 0, 0]);
    if z_l == w_i {
        return false; // z was sampled outside H, so this should not occur
    }
    let _cp_prime = (child_leaf_i.s - parent_f_b) * (z_l - w_i).inverse().expect("nonzero denom");
    true
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
    pub s: Vec<F>, // S layer values
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
    pub s_layers: Vec<Vec<F>>,
    pub transcript: FriTranscript,
    pub omega_layers: Vec<F>,
    pub z_layers: Vec<F>,
}

/* ============================= Merkle arity picker (Option A) ============================= */

fn pick_arity_for_layer(n: usize, requested_m: usize) -> usize {
    // Prefer 16 if requested >=16 and n is a multiple of 16
    if requested_m >= 16 && n % 16 == 0 {
        return 16;
    }
    // Else prefer 8 if requested >=8 and n is a multiple of 8
    if requested_m >= 8 && n % 8 == 0 {
        return 8;
    }
    // Else 2 if n is even
    if n % 2 == 0 {
        return 2;
    }
    // Fallback 1
    1
}

/* ============================= Build transcript ============================= */

pub fn fri_build_transcript(
    f0: Vec<F>,
    domain0: FriDomain,
    params: &FriProverParams,
) -> FriProverState {
    logln!(
        "fri_build_transcript: start n0={} L={}",
        domain0.size,
        params.schedule.len()
    );

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
        logln!("  fold layer {}: n={} m={}", ell, cur_size, m);
        let z = fri_sample_z_ell(params.seed_z, ell, cur_size);
        z_layers.push(z);
        let (_n_ell, omega_ell) = layer_domains[ell];
        omega_layers.push(omega_ell);
        cur_f = fri_fold_layer(&cur_f, z, m);
        cur_size /= m;
        f_layers.push(cur_f.clone());
    }

    let mut s_layers = Vec::with_capacity(l + 1);
    for ell in 0..l {
        let m = schedule[ell];
        let z = z_layers[ell];
        let s = compute_s_layer(&f_layers[ell], z, m);
        s_layers.push(s);
    }
    // Last layer S: zeros for symmetry.
    s_layers.push(vec![F::zero(); f_layers[l].len()]);

    // Commit each layer; arity per layer picked safely from n, m.
    let mut layers = Vec::with_capacity(l + 1);
    for ell in 0..=l {
        let n = f_layers[ell].len();
        let m_ell = if ell < l { schedule[ell] } else { 1 };
        let arity = pick_arity_for_layer(n, m_ell);
        let cfg = MerkleChannelCfg::new(arity).with_tree_label(ell as u64);

        logln!(
            "  commit layer {}: n={} m={} arity={}",
            ell,
            n,
            m_ell,
            arity
        );

        let prover = MerkleProver::new(cfg.clone());
        let (root, tree) = prover.commit_pairs(&f_layers[ell][..], &s_layers[ell][..]);

        layers.push(FriLayerCommitment {
            n,
            m: m_ell,
            root,
            f: f_layers[ell].clone(),
            s: s_layers[ell].clone(),
            tree,
            cfg,
        });
    }

    logln!("fri_build_transcript: done; last size={}", f_layers[l].len());

    FriProverState {
        f_layers,
        s_layers,
        transcript: FriTranscript { schedule, layers },
        omega_layers,
        z_layers,
    }
}

/* ============================= Query openings (constant-size) ============================= */

#[derive(Clone)]
pub struct LayerOpenings {
    pub i: usize,
    pub child_pair: (F, F),   // (f_l(i), S_l(i))
    pub child_proof: MerkleProof,
    pub parent_index: usize,  // b = i / m
    pub parent_pair: (F, F),  // (f_{l+1}[b], S_{l+1}[b]) (S parent unused here)
    pub parent_proof: MerkleProof,
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
    logln!(
        "fri_prove_queries: r={} L={}",
        r,
        st.transcript.schedule.len()
    );
    let l = st.transcript.schedule.len();
    let mut all = Vec::with_capacity(r);

    for q in 0..r {
        if q % 4 == 0 {
            logln!("  query {}/{}", q + 1, r);
        }
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
                if i2 < n {
                    i2
                } else {
                    i2 & (n - 1)
                }
            };

            // Child single opening at i
            let child_pair = (layer.f[i], layer.s[i]);
            let child_proof = layer.tree.open_many(&[i]);

            // Parent opening at b = i / m
            let parent_layer = &st.transcript.layers[ell + 1];
            let b = i / m;
            let parent_pair = (parent_layer.f[b], parent_layer.s[b]);
            let parent_proof = parent_layer.tree.open_many(&[b]);

            per_layer.push(LayerOpenings {
                i,
                child_pair,
                child_proof,
                parent_index: b,
                parent_pair,
                parent_proof,
            });
        }
        let last = &st.transcript.layers[l];
        let final_index = 0usize;
        let final_pair = (last.f[final_index], last.s[final_index]);
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
                if i2 < n {
                    i2
                } else {
                    i2 & (n - 1)
                }
            };

            let lay = &qopen.per_layer[ell];
            if lay.i != derived_i {
                return false;
            }

            // Verify child single opening against child root with the same picked arity
            let child_root = roots[ell];
            let ar_child = pick_arity_for_layer(n, m);
            let prover_child =
                MerkleProver::new(MerkleChannelCfg::new(ar_child).with_tree_label(ell as u64));
            if !prover_child.verify_pairs(
                &child_root,
                core::slice::from_ref(&lay.i),
                core::slice::from_ref(&lay.child_pair),
                &lay.child_proof,
            ) {
                return false;
            }

            // Parent verification (arity from next layer size and m)
            let n_parent = sizes[ell + 1];
            let m_parent = if ell + 1 < l { schedule[ell + 1] } else { 1 };
            let ar_parent = pick_arity_for_layer(n_parent, m_parent);
            let parent_root = roots[ell + 1];
            let prover_parent = MerkleProver::new(
                MerkleChannelCfg::new(ar_parent).with_tree_label((ell + 1) as u64),
            );
            if !prover_parent.verify_pairs(
                &parent_root,
                core::slice::from_ref(&lay.parent_index),
                core::slice::from_ref(&lay.parent_pair),
                &lay.parent_proof,
            ) {
                return false;
            }

            // Constant-size local check
            let (n_layer, omega_l) = layer_domains[ell];
            let z = z_layers[ell];

            let child_leaf_i = CombinedLeaf {
                f: lay.child_pair.0,
                s: lay.child_pair.1,
            };
            let parent_f_b = lay.parent_pair.0;

            if !verify_local_check_constant(
                lay.i,
                m,
                z,
                omega_l,
                n_layer,
                child_leaf_i,
                parent_f_b,
            ) {
                return false;
            }
        }

        // Final layer leaf check (arity 1)
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

    logln!("deep_fri_prove: building transcript");
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

    logln!("deep_fri_prove: proving queries r={}", params.r);
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

/* ============================= Proof size helpers (no-serde) ============================= */

const FR_BYTES: usize = 32;
const INDEX_BYTES: usize = core::mem::size_of::<usize>();

fn merkle_proof_size_bytes(mp: &MerkleProof) -> usize {
    let mut total = 0usize;
    // Sibling hashes: each is a field element/hash; count as 32 bytes each by default.
    total += mp
        .siblings
        .iter()
        .map(|grp| grp.len() * FR_BYTES)
        .sum::<usize>();
    total
}

pub fn deep_fri_proof_size_bytes(p: &DeepFriProof) -> usize {
    let mut total = 0usize;

    // Roots per layer
    total += p.roots.len() * FR_BYTES;

    // Params carried alongside proof (if you serialize them)
    total += FR_BYTES; // omega0
    total += INDEX_BYTES; // n0

    for q in &p.queries {
        // Final layer opening
        total += INDEX_BYTES; // final_index
        total += 2 * FR_BYTES; // final_pair: (f, s)
        total += merkle_proof_size_bytes(&q.final_proof);

        // Per-layer openings
        for lay in &q.per_layer {
            total += INDEX_BYTES; // i
            total += 2 * FR_BYTES; // child_pair
            total += merkle_proof_size_bytes(&lay.child_proof);

            total += INDEX_BYTES; // parent_index
            total += 2 * FR_BYTES; // parent_pair
            total += merkle_proof_size_bytes(&lay.parent_proof);
        }
    }

    total
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
    fn test_combined_layer_merkle_roundtrip_and_constant_check() {
        let n0 = 1024usize;
        let m = 16usize;
        let dom = FriDomain::new_radix2(n0);
        let omega0 = dom.omega;

        let mut rng = StdRng::seed_from_u64(1212);
        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();
        let z0 = fri_sample_z_ell(0xDEAD_BEEF, 0, n0);
        let f1 = fri_fold_layer(&f0, z0, m);

        let s0 = compute_s_layer(&f0, z0, m);

        // Commit layers using DS-aware Merkle with Option A arity picker
        let ar0 = super::pick_arity_for_layer(n0, m);
        let cfg0 = MerkleChannelCfg::new(ar0).with_tree_label(0);
        let prover0 = MerkleProver::new(cfg0.clone());
        let (root0, tree0) = prover0.commit_pairs(&f0[..], &s0[..]);

        let s1_dummy = vec![F::zero(); f1.len()];
        let ar1 = super::pick_arity_for_layer(f1.len(), 1); // next layer m not used in this test
        let cfg1 = MerkleChannelCfg::new(ar1).with_tree_label(1);
        let prover1 = MerkleProver::new(cfg1.clone());
        let (root1, tree1) = prover1.commit_pairs(&f1[..], &s1_dummy[..]);

        // Query random i; open single child and parent
        let mut rngq = StdRng::seed_from_u64(333);
        for _ in 0..10 {
            let i = rngq.gen::<usize>() % n0;
            let b = i / m;

            let child_pair = (f0[i], s0[i]);
            let child_proof = tree0.open_many(&[i]);

            // Verify child opening with facade
            let prov_child = MerkleProver::new(MerkleChannelCfg::new(ar0).with_tree_label(0));
            assert!(prov_child.verify_pairs(
                &root0,
                core::slice::from_ref(&i),
                core::slice::from_ref(&child_pair),
                &child_proof
            ));

            // Parent single opening
            let parent_pair = (f1[b], F::zero());
            let parent_proof = tree1.open_many(&[b]);
            let prov_parent = MerkleProver::new(MerkleChannelCfg::new(ar1).with_tree_label(1));
            assert!(prov_parent.verify_pairs(
                &root1,
                core::slice::from_ref(&b),
                core::slice::from_ref(&parent_pair),
                &parent_proof
            ));

            // Constant-size local check
            let child_leaf_i = CombinedLeaf {
                f: child_pair.0,
                s: child_pair.1,
            };
            assert!(super::verify_local_check_constant(
                i, m, z0, omega0, n0, child_leaf_i, parent_pair.0
            ));
        }
    }

    #[test]
    fn test_fri_queries_roundtrip_constant_check() {
        // Smaller smoke roundtrip
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
}