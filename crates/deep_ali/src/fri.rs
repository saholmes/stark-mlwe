use ark_ff::{Field, One, Zero};
use ark_pallas::Fr as F;
use ark_serialize::CanonicalSerialize;
use rand::{rngs::StdRng, Rng, SeedableRng};

use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
use ark_poly::EvaluationDomain;

use merkle::{MerkleChannelCfg, MerkleProof, MerkleProver, MerkleTree};
use transcript::{default_params as transcript_params, Transcript};

#[cfg(feature = "fri_bench_log")]
macro_rules! logln {
    ($($tt:tt)*) => { eprintln!($($tt)*); }
}
#[cfg(not(feature = "fri_bench_log"))]
macro_rules! logln {
    ($($tt:tt)*) => {};
}

mod ds {
    pub const FRI_SEED: &[u8] = b"FRI/seed";
    pub const FRI_INDEX: &[u8] = b"FRI/index";
    pub const FRI_Z_L: &[u8] = b"FRI/z/l";
    pub const FRI_LEAF: &[u8] = b"FRI/leaf";
}

fn tr_hash_fields_tagged(tag: &[u8], fields: &[F]) -> F {
    let mut tr = Transcript::new(b"FRI/FS", transcript_params());
    tr.absorb_bytes(tag);
    for &x in fields {
        tr.absorb_field(x);
    }
    tr.challenge(b"out")
}

// Poseidon hash of (f, s) to one field, domain-separated for leaves.
fn hash_leaf_pair(f: F, s: F) -> F {
    let mut tr = Transcript::new(b"FRI/leaf/poseidon", transcript_params());
    tr.absorb_bytes(ds::FRI_LEAF);
    tr.absorb_field(f);
    tr.absorb_field(s);
    tr.challenge(b"leaf")
}

#[derive(Clone, Copy, Debug)]
pub struct FriDomain {
    pub omega: F,
    pub size: usize,
}

impl FriDomain {
    pub fn new_radix2(size: usize) -> Self {
        let dom = Domain::<F>::new(size).expect("radix-2 domain exists");
        Self { omega: dom.group_gen, size }
    }
}

pub fn fri_sample_z_ell(seed_z: u64, level: usize, domain_size: usize) -> F {
    let fused = tr_hash_fields_tagged(
        ds::FRI_Z_L,
        &[F::from(seed_z), F::from(level as u64), F::from(domain_size as u64)],
    );
    let mut seed_bytes = [0u8; 32];
    fused.serialize_uncompressed(&mut seed_bytes[..]).expect("serialize");
    let mut rng = StdRng::from_seed(seed_bytes);

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
            if fallback.pow(&[domain_size as u64, 0, 0, 0]) != F::one() { return fallback; }
            return F::from(11u64);
        }
    }
}

pub fn fri_fold_layer(f_l: &[F], z_l: F, m: usize) -> Vec<F> {
    assert!(m >= 2);
    assert!(f_l.len() % m == 0, "layer size must be divisible by m");
    let n_next = f_l.len() / m;
    let mut out = vec![F::zero(); n_next];

    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m { z_pows.push(acc); acc *= z_l; }

    for b in 0..n_next {
        let base = b * m;
        let mut s = F::zero();
        for t in 0..m { s += f_l[base + t] * z_pows[t]; }
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
        assert!(cur_size % m == 0, "size must be divisible by m at level {level}");
        let z_l = fri_sample_z_ell(seed, level, cur_size);
        cur = fri_fold_layer(&cur, z_l, m);
        cur_size = cur.len();
        layers.push(cur.clone());
    }
    layers
}

#[derive(Clone, Copy, Debug)]
pub struct CombinedLeaf { pub f: F, pub s: F }

pub fn compute_s_layer(f_l: &[F], z_l: F, m: usize) -> Vec<F> {
    assert!(f_l.len() % m == 0);
    let n = f_l.len();
    let n_next = n / m;

    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m { z_pows.push(acc); acc *= z_l; }

    let mut s_bucket = vec![F::zero(); n_next];
    for b in 0..n_next {
        let base = b * m;
        let mut s = F::zero();
        for t in 0..m { s += f_l[base + t] * z_pows[t]; }
        s_bucket[b] = s;
    }

    let mut s_per_i = vec![F::zero(); n];
    for i in 0..n { s_per_i[i] = s_bucket[i / m]; }
    s_per_i
}

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

// Fold-consistency local check: enforce s_i == f_parent[b]
fn verify_local_check_fold(
    i: usize, m: usize, n_layer: usize,
    child_leaf_i: CombinedLeaf, parent_f_b: F,
) -> bool {
    let b = i / m;
    if b >= n_layer / m { return false; }
    child_leaf_i.s == parent_f_b
}

fn fs_seed_from_roots(roots: &[F]) -> F { tr_hash_fields_tagged(ds::FRI_SEED, roots) }

fn index_from_seed(seed_f: F, n_pow2: usize) -> usize {
    assert!(n_pow2.is_power_of_two());
    let mask = n_pow2 - 1;
    let mut seed_bytes = [0u8; 32];
    seed_f.serialize_uncompressed(&mut seed_bytes[..]).unwrap();
    let mut rng = StdRng::from_seed(seed_bytes);
    (rng.gen::<u64>() as usize) & mask
}

fn index_seed(roots_seed: F, ell: usize, q: usize) -> F {
    tr_hash_fields_tagged(ds::FRI_INDEX, &[roots_seed, F::from(ell as u64), F::from(q as u64)])
}

#[derive(Clone)]
pub struct FriLayerCommitment {
    pub n: usize,
    pub m: usize,
    pub root: F,
    pub f: Vec<F>,
    pub s: Vec<F>,
    pub hashed_leaves: bool, // true => single-column commit of Poseidon(f,s)
    pub tree: MerkleTree,
    pub cfg: MerkleChannelCfg,
}

#[derive(Clone)]
pub struct FriTranscript { pub schedule: Vec<usize>, pub layers: Vec<FriLayerCommitment> }

pub struct FriProverParams { pub schedule: Vec<usize>, pub seed_z: u64 }

pub struct FriProverState {
    pub f_layers: Vec<Vec<F>>,
    pub s_layers: Vec<Vec<F>>,
    pub transcript: FriTranscript,
    pub omega_layers: Vec<F>,
    pub z_layers: Vec<F>,
}

fn pick_arity_for_layer(n: usize, requested_m: usize) -> usize {
    if requested_m >= 16 && n % 16 == 0 { return 16; }
    if requested_m >= 8 && n % 8 == 0 { return 8; }
    if n % 2 == 0 { return 2; }
    1
}

pub fn fri_build_transcript(
    f0: Vec<F>, domain0: FriDomain, params: &FriProverParams,
) -> FriProverState {
    logln!("fri_build_transcript: start n0={} L={}", domain0.size, params.schedule.len());

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
    s_layers.push(vec![F::zero(); f_layers[l].len()]);

    let mut layers = Vec::with_capacity(l + 1);
    for ell in 0..=l {
        let n = f_layers[ell].len();
        let m_ell = if ell < l { schedule[ell] } else { 1 };
        let arity = pick_arity_for_layer(n, m_ell);
        let use_hashed = arity == 16 || arity == 8;

        let cfg = MerkleChannelCfg::new(arity).with_tree_label(ell as u64);
        let prover = MerkleProver::new(cfg.clone());

        let (root, tree) = if use_hashed {
            // True single-column commit of h = Poseidon(f, s)
            let mut h = Vec::with_capacity(n);
            for i in 0..n { h.push(hash_leaf_pair(f_layers[ell][i], s_layers[ell][i])); }
            let (root, tree) = prover.commit_single(&h[..]);
            logln!("  committed layer {}: n={} m={} arity={} hashed=1(single)", ell, n, m_ell, arity);
            (root, tree)
        } else {
            // For small arities, keep pair-commit of (f, s)
            let (root, tree) = prover.commit_pairs(&f_layers[ell][..], &s_layers[ell][..]);
            logln!("  committed layer {}: n={} m={} arity={} hashed=0(pairs)", ell, n, m_ell, arity);
            (root, tree)
        };

        layers.push(FriLayerCommitment {
            n, m: m_ell, root,
            f: f_layers[ell].clone(),
            s: s_layers[ell].clone(),
            hashed_leaves: use_hashed,
            tree, cfg,
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

// Per-layer batched multiproofs and per-query references.
#[derive(Clone)]
pub struct LayerBatchProof {
    pub hashed_leaves: bool,
    // Child batch
    pub child_indices: Vec<usize>, // unique, sorted
    pub child_proof: MerkleProof,  // union-of-paths multiproof
    // Parent batch
    pub parent_indices: Vec<usize>, // unique, sorted
    pub parent_proof: MerkleProof,  // union-of-paths multiproof
}

// For each query, reference positions in the batched arrays to avoid duplicating proofs.
#[derive(Clone)]
pub struct LayerQueryRef {
    pub i: usize,            // child leaf index
    pub child_pos: usize,    // position of i in child_indices
    pub parent_index: usize, // b = i / m
    pub parent_pos: usize,   // position of b in parent_indices
}

#[derive(Clone)]
pub struct FriQueryOpenings {
    pub per_layer_refs: Vec<LayerQueryRef>,
    pub final_index: usize,
    pub final_pair: (F, F),
}

#[derive(Clone)]
pub struct FriLayerBatches {
    pub layers: Vec<LayerBatchProof>,
    pub final_proof: MerkleProof, // final layer proof (usually trivial)
}

fn pos_in_sorted(xs: &[usize], x: usize) -> usize {
    match xs.binary_search(&x) {
        Ok(p) => p,
        Err(_) => unreachable!("index must be present in batch"),
    }
}

pub fn fri_prove_queries(
    st: &FriProverState, r: usize, roots_seed: F,
) -> (Vec<FriQueryOpenings>, Vec<F>, FriLayerBatches) {
    logln!("fri_prove_queries: r={} L={}", r, st.transcript.schedule.len());
    let L = st.transcript.schedule.len();

    // First, derive all indices per query per layer to enable batching.
    let mut all_refs: Vec<FriQueryOpenings> = Vec::with_capacity(r);
    let mut layer_child_buckets: Vec<Vec<usize>> = vec![Vec::new(); L];
    let mut layer_parent_buckets: Vec<Vec<usize>> = vec![Vec::new(); L];

    for q in 0..r {
        let mut per_layer_refs = Vec::with_capacity(L);
        for ell in 0..L {
            let layer = &st.transcript.layers[ell];
            let n = layer.n;
            let n_pow2 = n.next_power_of_two();
            let m = layer.m;

            let seed = index_seed(roots_seed, ell, q);
            let i0 = index_from_seed(seed, n_pow2);
            let i = if i0 < n {
                i0
            } else {
                let reseed = tr_hash_fields_tagged(ds::FRI_INDEX, &[seed, F::from(1u64)]);
                let i2 = index_from_seed(reseed, n_pow2);
                if i2 < n { i2 } else { i2 & (n - 1) }
            };
            let b = i / m;

            layer_child_buckets[ell].push(i);
            layer_parent_buckets[ell].push(b);

            per_layer_refs.push(LayerQueryRef {
                i,
                child_pos: 0, // fill after batching
                parent_index: b,
                parent_pos: 0, // fill after batching
            });
        }
        let last = &st.transcript.layers[L];
        all_refs.push(FriQueryOpenings {
            per_layer_refs,
            final_index: 0,
            final_pair: (last.f[0], last.s[0]),
        });
    }

    // Build batches per layer and fill positions.
    let mut layer_batches: Vec<LayerBatchProof> = Vec::with_capacity(L);
    for ell in 0..L {
        let lay = &st.transcript.layers[ell];

        // Child indices
        let mut child_idx = layer_child_buckets[ell].clone();
        child_idx.sort_unstable();
        child_idx.dedup();

        // Parent indices
        let mut parent_idx = layer_parent_buckets[ell].clone();
        parent_idx.sort_unstable();
        parent_idx.dedup();

        let child_proof = if lay.hashed_leaves {
            lay.tree.open_many_single(&child_idx)
        } else {
            lay.tree.open_many(&child_idx)
        };
        let parent_tree = &st.transcript.layers[ell + 1].tree;
        let parent_hashed = st.transcript.layers[ell + 1].hashed_leaves;
        let parent_proof = if parent_hashed {
            parent_tree.open_many_single(&parent_idx)
        } else {
            parent_tree.open_many(&parent_idx)
        };

        // Fill positions for all queries at this layer
        for q in 0..r {
            let rref = &mut all_refs[q].per_layer_refs[ell];
            rref.child_pos = pos_in_sorted(&child_idx, rref.i);
            rref.parent_pos = pos_in_sorted(&parent_idx, rref.parent_index);
            debug_assert_eq!(rref.parent_index, rref.i / lay.m);
        }

        layer_batches.push(LayerBatchProof {
            hashed_leaves: lay.hashed_leaves,
            child_indices: child_idx,
            child_proof,
            parent_indices: parent_idx,
            parent_proof,
        });
    }

    // Final layer proof (index 0)
    let last_layer = &st.transcript.layers[L];
    let last_indices = vec![0usize];
    let final_proof = if last_layer.hashed_leaves {
        last_layer.tree.open_many_single(&last_indices)
    } else {
        last_layer.tree.open_many(&last_indices)
    };

    let roots: Vec<F> = st.transcript.layers.iter().map(|l| l.root).collect();
    (
        all_refs,
        roots,
        FriLayerBatches {
            layers: layer_batches,
            final_proof,
        },
    )
}

// Public payload types

pub type AliA = Vec<F>;
pub type AliS = Vec<F>;
pub type AliE = Vec<F>;
pub type AliT = Vec<F>;

pub trait DeepAliBuilder {
    fn build_f0(&self, a: &AliA, s: &AliS, e: &AliE, t: &AliT, n0: usize, domain: FriDomain) -> Vec<F>;
}

#[derive(Clone, Default)]
pub struct DeepAliMock;

fn tr_hash_many(tag: &[u8], xs: &[F]) -> F { tr_hash_fields_tagged(tag, xs) }

impl DeepAliBuilder for DeepAliMock {
    fn build_f0(&self, a: &AliA, s: &AliS, e: &AliE, t: &AliT, n0: usize, _domain: FriDomain) -> Vec<F> {
        let seed_f = tr_hash_fields_tagged(
            b"ALI/mock/seed",
            &[tr_hash_many(b"ALI/a", a), tr_hash_many(b"ALI/s", s), tr_hash_many(b"ALI/e", e), tr_hash_many(b"ALI/t", t), F::from(n0 as u64)],
        );
        let mut seed_bytes = [0u8; 32];
        seed_f.serialize_uncompressed(&mut seed_bytes[..]).unwrap();
        let mut rng = StdRng::from_seed(seed_bytes);
        (0..n0).map(|_| F::from(rng.gen::<u64>())).collect()
    }
}

// Real DEEP-ALI builder using lib.rs merge helpers
pub struct DeepAliRealBuilder {
    pub r_eval_opt: Option<Vec<F>>, // optional blinding evaluations R on H
    pub use_blinding: bool,
    pub ds_tag: &'static [u8],       // domain-separation tag for (z, beta)
}

impl Default for DeepAliRealBuilder {
    fn default() -> Self {
        Self { r_eval_opt: None, use_blinding: false, ds_tag: b"ALI/DEEP" }
    }
}

// Deterministically derive (z, beta) for DEEP-ALI from a seed
fn ali_sample_z_beta_fs(tag: &[u8], n0: usize, roots_seed: F) -> (F, F) {
    let fused = tr_hash_fields_tagged(tag, &[roots_seed, F::from(n0 as u64)]);
    let mut seed_bytes = [0u8; 32];
    fused.serialize_uncompressed(&mut seed_bytes[..]).expect("serialize");
    let mut rng = StdRng::from_seed(seed_bytes);
    let beta = F::from(rng.gen::<u64>());
    let mut tries = 0usize;
    const MAX_TRIES: usize = 1_000;
    loop {
        let cand = F::from(rng.gen::<u64>());
        if !cand.is_zero() && cand.pow(&[n0 as u64, 0, 0, 0]) != F::one() {
            return (cand, beta);
        }
        tries += 1;
        if tries >= MAX_TRIES {
            let fallback = roots_seed + F::from(17u64);
            if fallback.pow(&[n0 as u64, 0, 0, 0]) != F::one() {
                return (fallback, beta);
            }
            return (F::from(19u64), beta);
        }
    }
}

impl DeepAliBuilder for DeepAliRealBuilder {
    fn build_f0(
        &self,
        a: &AliA, s: &AliS, e: &AliE, t: &AliT,
        n0: usize, domain: FriDomain,
    ) -> Vec<F> {
        use crate::{deep_ali_merge_evals, deep_ali_merge_evals_blinded};
        assert_eq!(a.len(), n0);
        assert_eq!(s.len(), n0);
        assert_eq!(e.len(), n0);
        assert_eq!(t.len(), n0);

        // FS-style seed from public ALI inputs
        let seed_f = tr_hash_fields_tagged(
            b"ALI/seed",
            &[
                tr_hash_fields_tagged(b"ALI/A", a),
                tr_hash_fields_tagged(b"ALI/S", s),
                tr_hash_fields_tagged(b"ALI/E", e),
                tr_hash_fields_tagged(b"ALI/T", t),
                F::from(n0 as u64),
            ],
        );

        let (z, beta) = ali_sample_z_beta_fs(self.ds_tag, n0, seed_f);
        let r_eval_opt_slice = self.r_eval_opt.as_ref().map(|v| &v[..]);

        let (f0_eval, _z_out, _c_star) = if self.use_blinding {
            deep_ali_merge_evals_blinded(a, s, e, t, r_eval_opt_slice, beta, domain.omega, z)
        } else {
            deep_ali_merge_evals(a, s, e, t, domain.omega, z)
        };

        f0_eval
    }
}

#[derive(Clone)]
pub struct LayerOpenPayload {
    pub f_i: F,
    pub s_i: F,
    pub f_parent_b: F,
    pub s_parent_b: F,
}

#[derive(Clone)]
pub struct FriQueryPayload {
    pub per_layer_refs: Vec<LayerQueryRef>,
    pub per_layer_payloads: Vec<LayerOpenPayload>,
    pub final_index: usize,
    pub final_pair: (F, F),
}

#[derive(Clone)]
pub struct DeepFriParams { pub schedule: Vec<usize>, pub r: usize, pub seed_z: u64 }

pub struct DeepFriProof {
    pub roots: Vec<F>,
    // Batched layer proofs shared across queries
    pub layer_batches: FriLayerBatches,
    // Per-query minimal refs and opened field elements
    pub queries: Vec<FriQueryPayload>,
    pub n0: usize,
    pub omega0: F,
}

pub fn deep_fri_prove<B: DeepAliBuilder>(
    builder: &B, a: &AliA, s: &AliS, e: &AliE, t: &AliT, n0: usize, params: &DeepFriParams,
) -> DeepFriProof {
    let domain0 = FriDomain::new_radix2(n0);
    let f0 = builder.build_f0(a, s, e, t, n0, domain0);

    logln!("deep_fri_prove: building transcript");
    let st = fri_build_transcript(
        f0, domain0,
        &FriProverParams { schedule: params.schedule.clone(), seed_z: params.seed_z },
    );

    let roots: Vec<F> = st.transcript.layers.iter().map(|l| l.root).collect();
    let roots_seed = fs_seed_from_roots(&roots);

    // Build batched openings and per-query refs
    let (refs_only, roots2, batches) = fri_prove_queries(&st, params.r, roots_seed);
    debug_assert_eq!(roots, roots2);

    // Assemble per-query field payloads so the verifier can recompute hashed and pair leaves in batch order
    let mut queries: Vec<FriQueryPayload> = Vec::with_capacity(params.r);
    for q in 0..params.r {
        let mut per_layer_payloads = Vec::with_capacity(params.schedule.len());
        for ell in 0..params.schedule.len() {
            let rref = &refs_only[q].per_layer_refs[ell];
            let f_i = st.transcript.layers[ell].f[rref.i];
            let s_i = st.transcript.layers[ell].s[rref.i];
            let f_parent_b = st.transcript.layers[ell + 1].f[rref.parent_index];
            let s_parent_b = st.transcript.layers[ell + 1].s[rref.parent_index];
            per_layer_payloads.push(LayerOpenPayload { f_i, s_i, f_parent_b, s_parent_b });
        }
        queries.push(FriQueryPayload {
            per_layer_refs: refs_only[q].per_layer_refs.clone(),
            per_layer_payloads,
            final_index: refs_only[q].final_index,
            final_pair: refs_only[q].final_pair,
        });
    }

    DeepFriProof { roots, layer_batches: batches, queries, n0, omega0: domain0.omega }
}

pub fn deep_fri_verify(params: &DeepFriParams, proof: &DeepFriProof) -> bool {
    let L = params.schedule.len();
    if proof.roots.len() != L + 1 { return false; }
    if proof.layer_batches.layers.len() != L { return false; }
    if proof.queries.len() != params.r { return false; }

    let sizes = layer_sizes_from_schedule(proof.n0, &params.schedule);

    // Prepare per-layer maps: index -> (f,s) and parent index -> (f,s)
    use std::collections::BTreeMap;
    let mut child_maps: Vec<BTreeMap<usize, (F, F)>> = vec![BTreeMap::new(); L];
    let mut parent_maps: Vec<BTreeMap<usize, (F, F)>> = vec![BTreeMap::new(); L];

    for q in 0..params.r {
        let qp = &proof.queries[q];
        if qp.per_layer_refs.len() != L || qp.per_layer_payloads.len() != L {
            return false;
        }
        for ell in 0..L {
            let rref = &qp.per_layer_refs[ell];
            let pay = &qp.per_layer_payloads[ell];
            child_maps[ell].entry(rref.i).or_insert((pay.f_i, pay.s_i));
            parent_maps[ell].entry(rref.parent_index).or_insert((pay.f_parent_b, pay.s_parent_b));
        }
    }

    // Verify per-layer child and parent batched multiproofs
    for ell in 0..L {
        let lb = &proof.layer_batches.layers[ell];

        // Child layer verification
        let ar_child = pick_arity_for_layer(sizes[ell], params.schedule[ell]);
        let hashed_child = ar_child == 16 || ar_child == 8;
        let prover_child = MerkleProver::new(MerkleChannelCfg::new(ar_child).with_tree_label(ell as u64));

        if hashed_child {
            let mut leaves_h = Vec::with_capacity(lb.child_indices.len());
            for &i in &lb.child_indices {
                let (f_i, s_i) = match child_maps[ell].get(&i) { Some(&p) => p, None => return false };
                leaves_h.push(hash_leaf_pair(f_i, s_i));
            }
            if !prover_child.verify_single(&proof.roots[ell], &lb.child_indices, &leaves_h, &lb.child_proof) {
                return false;
            }
        } else {
            let mut pairs = Vec::with_capacity(lb.child_indices.len());
            for &i in &lb.child_indices {
                let (f_i, s_i) = match child_maps[ell].get(&i) { Some(&p) => p, None => return false };
                pairs.push((f_i, s_i));
            }
            if !prover_child.verify_pairs(&proof.roots[ell], &lb.child_indices, &pairs, &lb.child_proof) {
                return false;
            }
        }

        // Parent layer verification (against root at ell+1)
        let ar_parent = pick_arity_for_layer(sizes[ell + 1], if ell + 1 < L { params.schedule[ell + 1] } else { 1 });
        let hashed_parent = ar_parent == 16 || ar_parent == 8;
        let prover_parent = MerkleProver::new(MerkleChannelCfg::new(ar_parent).with_tree_label((ell + 1) as u64));

        if hashed_parent {
            let mut leaves_parent_h = Vec::with_capacity(lb.parent_indices.len());
            for &b in &lb.parent_indices {
                let (fpb, spb) = match parent_maps[ell].get(&b) { Some(&p) => p, None => return false };
                leaves_parent_h.push(hash_leaf_pair(fpb, spb));
            }
            if !prover_parent.verify_single(&proof.roots[ell + 1], &lb.parent_indices, &leaves_parent_h, &lb.parent_proof) {
                return false;
            }
        } else {
            let mut pairs_parent = Vec::with_capacity(lb.parent_indices.len());
            for &b in &lb.parent_indices {
                let (fpb, spb) = match parent_maps[ell].get(&b) { Some(&p) => p, None => return false };
                pairs_parent.push((fpb, spb));
            }
            if !prover_parent.verify_pairs(&proof.roots[ell + 1], &lb.parent_indices, &pairs_parent, &lb.parent_proof) {
                return false;
            }
        }
    }

    // Local checks across queries: enforce s_i == f_parent[b]
    let layer_domains = layer_domains_from_schedule(proof.n0, &params.schedule);
    for q in 0..params.r {
        let qp = &proof.queries[q];
        for ell in 0..L {
            let rref = &qp.per_layer_refs[ell];
            let pay = &qp.per_layer_payloads[ell];
            let (n_layer, _omega_l) = layer_domains[ell];

            let child_leaf_i = CombinedLeaf { f: pay.f_i, s: pay.s_i };
            if !verify_local_check_fold(rref.i, params.schedule[ell], n_layer, child_leaf_i, pay.f_parent_b) {
                return false;
            }
        }
    }

    // Final layer proof: verify opening at index 0
    {
        let last_root = proof.roots[L];
        let ar_last = pick_arity_for_layer(sizes[L], 1);
        let hashed_last = ar_last == 16 || ar_last == 8;
        let prover_last = MerkleProver::new(MerkleChannelCfg::new(ar_last).with_tree_label(L as u64));
        let final_idx = proof.queries[0].final_index; // should be 0
        if final_idx != 0 { return false; }

        if hashed_last {
            let leaf_h = hash_leaf_pair(proof.queries[0].final_pair.0, proof.queries[0].final_pair.1);
            if !prover_last.verify_single(&last_root, &[final_idx], &[leaf_h], &proof.layer_batches.final_proof) {
                return false;
            }
        } else {
            if !prover_last.verify_pairs(&last_root, &[final_idx], &[proof.queries[0].final_pair], &proof.layer_batches.final_proof) {
                return false;
            }
        }
    }

    true
}

// ========== Proof size helpers (no-serde) ==========

const FR_BYTES: usize = 32;
const INDEX_BYTES: usize = core::mem::size_of::<usize>();

fn merkle_proof_size_bytes(mp: &MerkleProof) -> usize {
    let mut total = 0usize;
    total += mp.siblings.iter().map(|grp| grp.len() * FR_BYTES).sum::<usize>();
    total
}

pub fn deep_fri_proof_size_bytes(p: &DeepFriProof) -> usize {
    let mut total = 0usize;

    // Roots per layer
    total += p.roots.len() * FR_BYTES;

    // Params carried alongside proof (if you serialize them)
    total += FR_BYTES; // omega0
    total += INDEX_BYTES; // n0

    // Batched proofs per layer: child + parent
    for lb in &p.layer_batches.layers {
        total += merkle_proof_size_bytes(&lb.child_proof);
        total += merkle_proof_size_bytes(&lb.parent_proof);
        // Plus indices arrays (if serialized)
        total += lb.child_indices.len() * INDEX_BYTES;
        total += lb.parent_indices.len() * INDEX_BYTES;
    }
    total += merkle_proof_size_bytes(&p.layer_batches.final_proof);

    // Per-query small payloads
    for q in &p.queries {
        total += INDEX_BYTES; // final_index
        total += 2 * FR_BYTES; // final_pair
        // Per-layer refs and payloads
        total += q.per_layer_refs.len() * (2 * INDEX_BYTES); // child_pos + parent_pos
        total += q.per_layer_payloads.len() * (4 * FR_BYTES); // f_i, s_i, f_parent_b, s_parent_b
    }

    total
}