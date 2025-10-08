//! DEEP-ALI + DEEP-FRI with Poseidon-based Merkle and FS, preserving Milestone 8 semantics.

use ark_ff::{Field, One, PrimeField, Zero};
use ark_pallas::Fr as F;
use ark_serialize::CanonicalSerialize;
use rand::{rngs::StdRng, Rng, SeedableRng};

use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
use ark_poly::EvaluationDomain;

// Poseidon
use ark_sponge::{poseidon::PoseidonConfig, poseidon::PoseidonSponge, CryptographicSponge};
use std::sync::OnceLock;

/* ============================== Domain separation ============================== */

mod ds {
    pub const FRI_SEED: &str = "FRI/seed";
    pub const FRI_INDEX: &str = "FRI/index";
    pub const FRI_Z_L: &str = "FRI/z/l";

    pub const MT_LEAF: &str = "MT/leaf";
    pub const MT_NODE_ARITY2_LEVEL: &str = "MT/node/arity=2/level=";
}

/* ======================= Poseidon configuration (width=3) ======================= */

const POSEIDON_FULL_ROUNDS: usize = 8;
const POSEIDON_PARTIAL_ROUNDS: usize = 57;
const POSEIDON_ALPHA: u64 = 5;
const POSEIDON_RATE: usize = 2;
const POSEIDON_CAPACITY: usize = 1;
const POSEIDON_WIDTH: usize = POSEIDON_RATE + POSEIDON_CAPACITY;

fn poseidon_mds() -> Vec<Vec<F>> {
    // Fixed 3x3 matrix, reduced from bytes.
    let data: [[&[u8; 32]; 3]; 3] = [
        [
            b"\x9b\x9f\x0f\x9d\x3a\x2e\x6c\x98\xbc\x1a\x7d\x16\x3d\x32\x21\x20\xc5\x9e\xaf\x7d\xbf\x8f\x6b\x3f\x8b\xf1\x3c\x6c\x4e\x2d\x9f\x01",
            b"\x2d\x7f\x98\xb1\xcb\x56\x7e\x8d\x0b\x1e\x8c\x2f\xac\x3b\x4e\x0c\xee\x44\x93\xa4\xbf\x0a\x5d\x6b\x9c\x20\x33\x59\x77\xaa\xcc\x11",
            b"\x7f\x06\x57\x2e\x0a\x77\x3f\x55\x56\x8a\x2b\x8b\x7d\x33\x0d\x88\x81\x33\x22\x47\xd1\x9b\x77\xf1\x33\x05\x66\x90\x0a\xf7\x55\x21",
        ],
        [
            b"\x53\x2f\x43\x18\x11\xaa\x0d\x65\x72\xac\x0e\x99\x25\x61\x73\x61\x2a\xaf\x3f\x61\x6d\x3a\x44\xf9\x86\x6d\x3a\x5d\xfb\xa1\xbb\x09",
            b"\xe1\xb5\x22\x4b\xf1\x3d\x77\x7a\x77\x77\x45\x4d\xcb\xee\xab\x72\x55\x08\x24\xee\x10\x9e\xc8\x88\x1b\x55\x42\xba\x8c\x98\x01\x02",
            b"\x0b\x3d\x28\x24\x1c\xcd\xaa\x8a\xfe\x66\x3a\x31\x49\x0c\x8a\x21\x41\x1a\x7c\x3e\xef\x77\x8f\x1a\x1d\x15\x3b\xbb\x61\x22\x33\x44",
        ],
        [
            b"\x8a\x49\x88\x2b\x4c\x3a\x8b\x01\x1f\xc2\x5c\x4a\x9e\xaa\x28\x55\x77\x12\xff\x01\x48\xaa\x3e\xa7\xcf\xd2\x51\x77\x11\x81\x1a\x0f",
            b"\x2a\x41\x2f\x77\xcc\x9d\x02\x3a\x55\x72\x12\x48\x7a\x1a\x1a\x3b\x44\x0c\x9a\x01\xc1\xbb\x99\x0d\xde\x22\x6e\x01\x42\x57\x91\xfe",
            b"\x01\x9a\x2b\x0f\xcd\xaa\x77\x91\x0d\x44\x88\x3b\x1a\x5e\x01\x72\xb3\x91\x77\x01\x7a\xcc\x31\x55\x99\x12\xfe\x09\x30\x44\x17\x7c",
        ],
    ];
    let mut m = vec![vec![F::zero(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = F::from_le_bytes_mod_order(data[i][j]);
        }
    }
    m
}

fn poseidon_ark() -> Vec<Vec<F>> {
    // Deterministic ARK expansion from seed (fixed; no RNG at runtime).
    let mut rng = StdRng::seed_from_u64(0xC0FFEE_BABE_1234);
    let rounds = POSEIDON_FULL_ROUNDS + POSEIDON_PARTIAL_ROUNDS;
    let mut ark = vec![vec![F::zero(); POSEIDON_WIDTH]; rounds];
    for r in 0..rounds {
        for c in 0..POSEIDON_WIDTH {
            let mut buf = [0u8; 32];
            rng.fill(&mut buf);
            ark[r][c] = F::from_le_bytes_mod_order(&buf);
        }
    }
    ark
}

static POSEIDON_CONFIG: OnceLock<PoseidonConfig<F>> = OnceLock::new();

fn poseidon_sponge() -> PoseidonSponge<F> {
    let cfg = POSEIDON_CONFIG.get_or_init(|| {
        PoseidonConfig::<F>::new(
            POSEIDON_FULL_ROUNDS,
            POSEIDON_PARTIAL_ROUNDS,
            POSEIDON_ALPHA,
            poseidon_mds(),
            poseidon_ark(),
            POSEIDON_RATE,
            POSEIDON_CAPACITY,
        )
    });
    PoseidonSponge::<F>::new(cfg)
}

fn poseidon_hash_fields_tagged(tag: &str, fields: &[F]) -> F {
    let mut s = poseidon_sponge();
    s.absorb(&F::from_le_bytes_mod_order(tag.as_bytes()));
    s.absorb(&fields.to_vec());
    s.squeeze_field_elements(1)[0]
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
        Self { omega: dom.group_gen, size }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoseidonParamsPlaceholder;

#[derive(Clone, Debug)]
pub struct FriLayerParams {
    pub m: usize,
    pub width: usize,
    pub poseidon_params: PoseidonParamsPlaceholder,
    pub level: usize,
}

impl FriLayerParams {
    pub fn new(level: usize, m: usize) -> Self {
        Self { m, width: m, poseidon_params: PoseidonParamsPlaceholder::default(), level }
    }
}

/* ============================== FS challenges ============================== */

pub fn fri_sample_z_ell(seed_z: u64, level: usize, domain_size: usize) -> F {
    // Poseidon FS-based sampling, uniform over field and reject if in domain H (z^N = 1).
    let fused = poseidon_hash_fields_tagged(ds::FRI_Z_L, &[F::from(seed_z), F::from(level as u64), F::from(domain_size as u64)]);
    let mut seed_bytes = [0u8; 32];
    fused.serialize_uncompressed(&mut seed_bytes[..]).expect("ser");
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
        assert!(cur_size % m == 0, "size must be divisible by m at level {level}");
        let z_l = fri_sample_z_ell(seed, level, cur_size);
        cur = fri_fold_layer(&cur, z_l, m);
        cur_size = cur.len();
        layers.push(cur.clone());
    }
    layers
}

/* ========================== Legacy single-value commit ========================== */

#[derive(Clone, Debug)]
pub struct FriCommitment {
    pub digests: Vec<F>,
}

impl FriCommitment {
    pub fn commit(values: &[F]) -> Self {
        let digests = values.iter().map(|v| poseidon_hash_fields_tagged("FRI/legacy/leaf", &[*v])).collect();
        Self { digests }
    }
    pub fn open(&self, idx: usize) -> F { self.digests[idx] }
    pub fn verify(&self, idx: usize, v: F) -> bool {
        poseidon_hash_fields_tagged("FRI/legacy/leaf", &[v]) == self.digests[idx]
    }
}

/* ============================== Combined leaf type ============================== */

#[derive(Clone, Copy, Debug)]
pub struct CombinedLeaf {
    pub f: F,
    pub cp: F,
}

/* =========================== Poseidon Merkle (arity=2) =========================== */

#[derive(Clone)]
pub struct PoseidonMerkle {
    pub root: F,
    pub nodes: Vec<F>, // 1-indexed, 0 unused
    pub n_leaves_pow2: usize,
}

#[derive(Clone, Copy)]
pub enum Dir { Left, Right }

#[derive(Clone)]
pub struct MerkleProof {
    pub siblings: Vec<F>,
    pub dirs: Vec<Dir>,
    pub leaf_index: usize,
}

fn mt_leaf_hash(leaf: &CombinedLeaf) -> F {
    poseidon_hash_fields_tagged(ds::MT_LEAF, &[leaf.f, leaf.cp])
}

fn mt_node_hash(level: usize, left: F, right: F) -> F {
    let tag = format!("{}{}", ds::MT_NODE_ARITY2_LEVEL, level);
    poseidon_hash_fields_tagged(&tag, &[left, right])
}

impl PoseidonMerkle {
    pub fn build(leaves: &[CombinedLeaf]) -> Self {
        assert!(!leaves.is_empty());
        let n = leaves.len();
        let n_pow2 = n.next_power_of_two();

        // Level 0: leaf hashes
        let mut level_nodes = Vec::with_capacity(n_pow2);
        let zero_leaf = CombinedLeaf { f: F::zero(), cp: F::zero() };
        let zero_hash = mt_leaf_hash(&zero_leaf);
        for i in 0..n_pow2 {
            level_nodes.push(if i < n { mt_leaf_hash(&leaves[i]) } else { zero_hash });
        }

        // Store all levels (from leaves up to root), with level indices starting at 0 above leaves
        let mut all_levels: Vec<Vec<F>> = Vec::new();
        all_levels.push(level_nodes.clone()); // leaves (treated as level -1 conceptually)

        let mut cur = level_nodes;
        let mut level = 0usize;
        while cur.len() > 1 {
            let mut next = Vec::with_capacity(cur.len() / 2);
            for k in 0..(cur.len() / 2) {
                next.push(mt_node_hash(level, cur[2 * k], cur[2 * k + 1]));
            }
            all_levels.push(next.clone());
            cur = next;
            level += 1;
        }
        let root = all_levels.last().unwrap()[0];

        // Build 1-indexed flat nodes array:
        // Leaves at [n_pow2 .. 2*n_pow2-1], internal nodes above; but we only need sibling hashes for open().
        // We'll fill leaves, then for internal nodes recompute using children with the same level tags.
        let total_nodes = 2 * n_pow2;
        let mut nodes = vec![F::zero(); total_nodes];
        // Fill leaves
        for i in 0..n_pow2 {
            nodes[n_pow2 + i] = all_levels[0][i];
        }
        // Fill internal nodes upward with correct levels
        let mut width = n_pow2;
        let mut idx_start = n_pow2 / 2; // number of nodes at the first internal level
        let mut level_tag = 0usize;
        while width > 1 {
            for k in 0..(width / 2) {
                let parent_idx = idx_start + k;
                let left_idx = 2 * parent_idx;
                let right_idx = 2 * parent_idx + 1;
                // Children are already filled
                let left = nodes[left_idx];
                let right = nodes[right_idx];
                nodes[parent_idx] = mt_node_hash(level_tag, left, right);
            }
            width /= 2;
            idx_start /= 2;
            level_tag += 1;
        }

        Self { root, nodes, n_leaves_pow2: n_pow2 }
    }

    pub fn open(&self, leaf_idx: usize) -> MerkleProof {
        assert!(leaf_idx < self.n_leaves_pow2);
        let mut idx = self.n_leaves_pow2 + leaf_idx;
        let mut siblings = Vec::new();
        let mut dirs = Vec::new();
        while idx > 1 {
            let is_right = (idx % 2) == 1;
            let sib_idx = if is_right { idx - 1 } else { idx + 1 };
            siblings.push(self.nodes[sib_idx]);
            dirs.push(if is_right { Dir::Right } else { Dir::Left });
            idx /= 2;
        }
        MerkleProof { siblings, dirs, leaf_index: leaf_idx }
    }

    pub fn verify_leaf(root: F, leaf: &CombinedLeaf, proof: &MerkleProof, n_leaves_pow2: usize) -> bool {
        // Replay the same level tags: combine siblings starting at level 0
        let mut cur = mt_leaf_hash(leaf);
        let mut level = 0usize;
        for (sib, dir) in proof.siblings.iter().zip(proof.dirs.iter()) {
            cur = match dir {
                Dir::Right => mt_node_hash(level, *sib, cur),
                Dir::Left => mt_node_hash(level, cur, *sib),
            };
            level += 1;
        }
        cur == root
    }
}

#[derive(Clone)]
pub struct CombinedPoseidonCommitment {
    pub root: F,
    pub merkle: PoseidonMerkle,
    pub leaves_len: usize,
}

impl CombinedPoseidonCommitment {
    pub fn commit(values: &[CombinedLeaf]) -> Self {
        let merkle = PoseidonMerkle::build(values);
        Self { root: merkle.root, merkle, leaves_len: values.len() }
    }
    pub fn open(&self, i: usize) -> MerkleProof { self.merkle.open(i) }
    pub fn verify(&self, i: usize, value: CombinedLeaf, proof: &MerkleProof) -> bool {
        proof.leaf_index == i && PoseidonMerkle::verify_leaf(self.root, &value, proof, self.merkle.n_leaves_pow2)
    }
}

/* ============================= CP computation ============================= */

pub fn compute_cp_layer(
    f_l: &[F],
    f_l_plus_1: &[F],
    z_l: F,
    m: usize,
    omega_l: F,
) -> Vec<F> {
    assert!(f_l.len() % m == 0, "size must be divisible by m");
    let n = f_l.len();
    let n_next = n / m;
    assert_eq!(f_l_plus_1.len(), n_next, "f_l_plus_1 length mismatch");

    // Precompute z^t (t=0..m-1)
    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m { z_pows.push(acc); acc *= z_l; }

    // Precompute omega^i (i=0..n-1), natural order
    let mut omega_pows = Vec::with_capacity(n);
    let mut w = F::one();
    for _ in 0..n { omega_pows.push(w); w *= omega_l; }

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
    f_l.iter().zip(cp_l.iter()).map(|(&f, &cp)| CombinedLeaf { f, cp }).collect()
}

/* ============================= Helpers: sizes and domains ============================= */

fn layer_sizes_from_schedule(n0: usize, schedule: &[usize]) -> Vec<usize> {
    let mut sizes = Vec::with_capacity(schedule.len() + 1);
    let mut n = n0;
    sizes.push(n);
    for &m in schedule {
        assert!(n % m == 0);
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

/* ============================= Neighbor openings ============================= */

pub fn open_bucket_neighbors(
    commitment: &CombinedPoseidonCommitment,
    leaves: &[CombinedLeaf],
    i: usize,
    m: usize,
) -> (Vec<usize>, Vec<CombinedLeaf>, Vec<MerkleProof>) {
    let n = leaves.len();
    assert!(n % m == 0);
    let b = i / m;
    let base = b * m;
    let mut idxs = Vec::with_capacity(m - 1);
    let mut vals = Vec::with_capacity(m - 1);
    let mut proofs = Vec::with_capacity(m - 1);
    for t in 0..m {
        let j = base + t;
        if j == i { continue; }
        idxs.push(j);
        vals.push(leaves[j]);
        proofs.push(commitment.open(j));
    }
    (idxs, vals, proofs)
}

/* ============================= Local check verify ============================= */

pub fn verify_local_check_with_openings(
    child_commit: &CombinedPoseidonCommitment,
    parent_commit: &CombinedPoseidonCommitment,
    i: usize,
    leaf_i: CombinedLeaf,
    proof_i: &MerkleProof,
    neighbor_indices: &[usize],
    neighbor_leaves: &[CombinedLeaf],
    neighbor_proofs: &[MerkleProof],
    parent_idx: usize,
    parent_leaf: CombinedLeaf,
    parent_proof: &MerkleProof,
    z_l: F,
    m: usize,
    omega_l: F,
    n_layer: usize,
) -> bool {
    // 1) Merkle inclusion: child i, neighbors, and parent
    if !child_commit.verify(i, leaf_i, proof_i) {
        #[cfg(test)]
        panic!("Child Merkle verification failed at i={}", i);
        #[cfg(not(test))]
        return false;
    }
    if neighbor_indices.len() != neighbor_leaves.len()
        || neighbor_indices.len() != neighbor_proofs.len()
    {
        #[cfg(test)]
        panic!("Neighbor vector length mismatch");
        #[cfg(not(test))]
        return false;
    }
    for ((&j, &leaf_j), pf) in neighbor_indices
        .iter()
        .zip(neighbor_leaves.iter())
        .zip(neighbor_proofs.iter())
    {
        if pf.leaf_index != j {
            #[cfg(test)]
            panic!("Neighbor proof index mismatch: pf.leaf_index={} j={}", pf.leaf_index, j);
            #[cfg(not(test))]
            return false;
        }
        if !child_commit.verify(j, leaf_j, pf) {
            #[cfg(test)]
            panic!("Neighbor Merkle verification failed at j={}", j);
            #[cfg(not(test))]
            return false;
        }
    }
    if !parent_commit.verify(parent_idx, parent_leaf, parent_proof) {
        #[cfg(test)]
        panic!("Parent Merkle verification failed at parent_idx={}", parent_idx);
        #[cfg(not(test))]
        return false;
    }

    // 2) Bucket and equation
    let b = i / m;
    if parent_idx != b {
        #[cfg(test)]
        panic!("Parent index mismatch: parent_idx={} expected={}", parent_idx, b);
        #[cfg(not(test))]
        return false;
    }

    // Strict neighbor set validation
    let base = b * m;
    #[cfg(test)]
    {
        use std::collections::BTreeSet;
        let expected: BTreeSet<_> = (0..m).map(|t| base + t).filter(|&j| j != i).collect();
        let actual: BTreeSet<_> = neighbor_indices.iter().copied().collect();
        if expected != actual {
            panic!(
                "Neighbor index set mismatch: expected={:?} actual={:?} (base={} m={} i={})",
                expected, actual, base, m, i
            );
        }
    }

    // Gather fs in canonical order
    let mut fs = vec![F::zero(); m];
    for (&j, &leaf_j) in neighbor_indices.iter().zip(neighbor_leaves.iter()) {
        let t = j - base;
        if t >= m {
            #[cfg(test)]
            panic!("Neighbor out of bucket: j={} base={} m={}", j, base, m);
            #[cfg(not(test))]
            return false;
        }
        fs[t] = leaf_j.f;
    }
    let t_i = i - base;
    if t_i >= m {
        #[cfg(test)]
        panic!("i out of bucket: i={} base={} m={}", i, base, m);
        #[cfg(not(test))]
        return false;
    }
    fs[t_i] = leaf_i.f;

    // RHS: Σ fs[t] z^t
    let mut rhs = F::zero();
    let mut pow = F::one();
    for t in 0..m {
        rhs += fs[t] * pow;
        pow *= z_l;
    }

    // ω^i with i reduced modulo the logical layer size
    let w_i = omega_l.pow(&[(i % n_layer) as u64, 0, 0, 0]);

    // LHS: cp(i)*(z - w_i) + f_{l+1}(b)
    let lhs = leaf_i.cp * (z_l - w_i) + parent_leaf.f;

    if lhs != rhs {
        // Force diagnostic
        let _ser = |x: F| {
            let mut buf = [0u8; 32];
            x.serialize_uncompressed(&mut buf[..]).ok();
            buf
        };
        let mut rhs_check = F::zero();
        let mut pow2 = F::one();
        for t in 0..m {
            rhs_check += fs[t] * pow2;
            pow2 *= z_l;
        }
        #[cfg(test)]
        {
            panic!(
                "[FRI mismatch] i={} base={} m={} N={} t_i={} lhs!=rhs",
                i, base, m, n_layer, t_i
            );
        }
        #[cfg(not(test))]
        return false;
    }

    true
}

/* ============================= FS helpers ============================= */

fn fs_seed_from_roots(roots: &[F]) -> F {
    poseidon_hash_fields_tagged(ds::FRI_SEED, roots)
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
    poseidon_hash_fields_tagged(ds::FRI_INDEX, &[roots_seed, F::from(ell as u64), F::from(q as u64)])
}

/* ============================= Transcript structures ============================= */

#[derive(Clone)]
pub struct FriLayerCommitment {
    pub leaves: Vec<CombinedLeaf>,
    pub com: CombinedPoseidonCommitment,
    pub root: F,
    pub n: usize,
    pub m: usize,
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

    let mut layers = Vec::with_capacity(l + 1);
    for ell in 0..=l {
        let leaves = build_combined_layer(&f_layers[ell], &cp_layers[ell]);
        let com = CombinedPoseidonCommitment::commit(&leaves);
        layers.push(FriLayerCommitment {
            n: leaves.len(),
            m: if ell < l { schedule[ell] } else { 1 },
            root: com.root,
            com,
            leaves,
        });
    }

    FriProverState { f_layers, cp_layers, transcript: FriTranscript { schedule, layers }, omega_layers, z_layers }
}

/* ============================= Query openings ============================= */

#[derive(Clone)]
pub struct LayerOpenings {
    pub i: usize,
    pub leaf_i: CombinedLeaf,
    pub proof_i: MerkleProof,
    pub neighbor_idx: Vec<usize>,
    pub neighbor_leaves: Vec<CombinedLeaf>,
    pub neighbor_proofs: Vec<MerkleProof>,
    pub parent_idx: usize,
    pub parent_leaf: CombinedLeaf,
    pub parent_proof: MerkleProof,
}

#[derive(Clone)]
pub struct FriQueryOpenings {
    pub per_layer: Vec<LayerOpenings>,
    pub final_leaf: CombinedLeaf,
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
            let n_pow2 = layer.n.next_power_of_two();

            let seed = index_seed(roots_seed, ell, q);
            let i = index_from_seed(seed, n_pow2);
            if i >= layer.n {
                let reseed = poseidon_hash_fields_tagged(ds::FRI_INDEX, &[seed, F::from(1u64)]);
                let i2 = index_from_seed(reseed, n_pow2);
                let i = if i2 < layer.n { i2 } else { i2 & (layer.n - 1) };
                let parent_idx = i / layer.m;
                let parent_layer = &st.transcript.layers[ell + 1];
                let leaf_i = layer.leaves[i];
                let proof_i = layer.com.open(i);
                let (neighbor_idx, neighbor_leaves, neighbor_proofs) = open_bucket_neighbors(&layer.com, &layer.leaves, i, layer.m);
                let parent_leaf = parent_layer.leaves[parent_idx];
                let parent_proof = parent_layer.com.open(parent_idx);
                per_layer.push(LayerOpenings { i, leaf_i, proof_i, neighbor_idx, neighbor_leaves, neighbor_proofs, parent_idx, parent_leaf, parent_proof });
                continue;
            }

            let parent_idx = i / layer.m;
            let parent_layer = &st.transcript.layers[ell + 1];
            let leaf_i = layer.leaves[i];
            let proof_i = layer.com.open(i);
            let (neighbor_idx, neighbor_leaves, neighbor_proofs) =
                open_bucket_neighbors(&layer.com, &layer.leaves, i, layer.m);
            let parent_leaf = parent_layer.leaves[parent_idx];
            let parent_proof = parent_layer.com.open(parent_idx);
            per_layer.push(LayerOpenings { i, leaf_i, proof_i, neighbor_idx, neighbor_leaves, neighbor_proofs, parent_idx, parent_leaf, parent_proof });
        }
        let last = &st.transcript.layers[l];
        let final_leaf = last.leaves[0];
        let final_proof = last.com.open(0);
        all.push(FriQueryOpenings { per_layer, final_leaf, final_proof });
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
    if roots.len() != l + 1 { return false; }

    let sizes = layer_sizes_from_schedule(n0, schedule);
    let layer_domains = layer_domains_from_schedule(n0, schedule);

    // z_ell from FS seed
    let mut z_layers = Vec::with_capacity(l);
    for ell in 0..l {
        z_layers.push(fri_sample_z_ell(seed_z, ell, sizes[ell]));
    }

    let roots_seed = fs_seed_from_roots(roots);

    for (q, qopen) in queries.iter().enumerate().take(r) {
        if qopen.per_layer.len() != l { return false; }

        for ell in 0..l {
            let lay = &qopen.per_layer[ell];
            let n = sizes[ell];
            let n_pow2 = n.next_power_of_two();

            let seed = index_seed(roots_seed, ell, q);
            let i = index_from_seed(seed, n_pow2);
            let derived_i = if i < n {
                i
            } else {
                let reseed = poseidon_hash_fields_tagged(ds::FRI_INDEX, &[seed, F::from(1u64)]);
                let i2 = index_from_seed(reseed, n_pow2);
                if i2 < n { i2 } else { i2 & (n - 1) }
            };

            if lay.proof_i.leaf_index != derived_i || lay.i != derived_i {
                return false;
            }

            if !PoseidonMerkle::verify_leaf(roots[ell], &lay.leaf_i, &lay.proof_i, n_pow2) {
                return false;
            }
            for ((&j, leaf_j), pf) in lay.neighbor_idx.iter().zip(lay.neighbor_leaves.iter()).zip(lay.neighbor_proofs.iter()) {
                if pf.leaf_index != j { return false; }
                if !PoseidonMerkle::verify_leaf(roots[ell], leaf_j, pf, n_pow2) { return false; }
            }

            let expected_parent_idx = derived_i / schedule[ell];
            if lay.parent_idx != expected_parent_idx {
                return false;
            }
            let n_pow2_p = sizes[ell + 1].next_power_of_two();
            if !PoseidonMerkle::verify_leaf(roots[ell + 1], &lay.parent_leaf, &lay.parent_proof, n_pow2_p) {
                return false;
            }

            let z = z_layers[ell];
            let (n_layer, omega_l) = layer_domains[ell];

            let child_commit = CombinedPoseidonCommitment {
                root: roots[ell],
                merkle: PoseidonMerkle { root: roots[ell], nodes: vec![], n_leaves_pow2: n_pow2 },
                leaves_len: 0,
            };
            let parent_commit = CombinedPoseidonCommitment {
                root: roots[ell + 1],
                merkle: PoseidonMerkle { root: roots[ell + 1], nodes: vec![], n_leaves_pow2: n_pow2_p },
                leaves_len: 0,
            };

            let ok = verify_local_check_with_openings(
                &child_commit,
                &parent_commit,
                derived_i,
                lay.leaf_i,
                &lay.proof_i,
                &lay.neighbor_idx,
                &lay.neighbor_leaves,
                &lay.neighbor_proofs,
                expected_parent_idx,
                lay.parent_leaf,
                &lay.parent_proof,
                z,
                schedule[ell],
                omega_l,
                n_layer,
            );
            if !ok { return false; }
        }

        let n_pow2_l = sizes[l].next_power_of_two();
        if !PoseidonMerkle::verify_leaf(roots[l], &qopen.final_leaf, &qopen.final_proof, n_pow2_l) {
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
    fn build_f0(&self, a: &AliA, s: &AliS, e: &AliE, t: &AliT, n0: usize, domain: FriDomain) -> Vec<F>;
}

#[derive(Clone, Default)]
pub struct DeepAliMock;

impl DeepAliBuilder for DeepAliMock {
    fn build_f0(&self, a: &AliA, s: &AliS, e: &AliE, t: &AliT, n0: usize, _domain: FriDomain) -> Vec<F> {
        let seed_f = poseidon_hash_fields_tagged(
            "ALI/mock/seed",
            &[
                poseidon_hash_fields_tagged("ALI/a", a),
                poseidon_hash_fields_tagged("ALI/s", s),
                poseidon_hash_fields_tagged("ALI/e", e),
                poseidon_hash_fields_tagged("ALI/t", t),
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

    let st = fri_build_transcript(f0, domain0, &FriProverParams { schedule: params.schedule.clone(), seed_z: params.seed_z });

    let roots: Vec<F> = st.transcript.layers.iter().map(|l| l.root).collect();
    let roots_seed = fs_seed_from_roots(&roots);
    let (queries, roots2) = fri_prove_queries(&st, params.r, roots_seed);
    debug_assert_eq!(roots, roots2);

    DeepFriProof { roots, queries, n0, omega0: domain0.omega }
}

pub fn deep_fri_verify(params: &DeepFriParams, proof: &DeepFriProof) -> bool {
    fri_verify_queries(&params.schedule, proof.n0, proof.omega0, params.seed_z, &proof.roots, &proof.queries, params.r)
}

/* ===================================== Tests ===================================== */

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
    fn test_layer_commit_open_basic() {
        let n0 = 4096usize;
        let schedule = [16usize, 16usize, 8usize];
        let mut rng = StdRng::seed_from_u64(8888);
        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();
        let layers = fri_fold_schedule(f0.clone(), &schedule, 0xBAD5EED);
        for (ell, f) in layers.iter().enumerate() {
            let com = FriCommitment::commit(f);
            let mut rngp = StdRng::seed_from_u64(ell as u64 * 101);
            for _ in 0..5 {
                let idx = (rngp.gen::<usize>() % f.len()).min(f.len().saturating_sub(1));
                assert!(com.verify(idx, f[idx]));
                assert_eq!(com.open(idx), com.digests[idx]);
            }
        }
    }

    #[test]
    fn test_combined_layer_poseidon_merkle_roundtrip_and_local_verify() {
        let n0 = 1024usize;
        let m = 16usize;
        let dom = FriDomain::new_radix2(n0);
        let omega0 = dom.omega;

        let mut rng = StdRng::seed_from_u64(1212);
        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();
        let z0 = fri_sample_z_ell(0xDEAD_BEEF, 0, n0);
        let f1 = fri_fold_layer(&f0, z0, m);

        let cp0 = compute_cp_layer(&f0, &f1, z0, m, omega0);

        let leaves0 = build_combined_layer(&f0, &cp0);
        let com0 = CombinedPoseidonCommitment::commit(&leaves0);

        let cp1_dummy = vec![F::zero(); f1.len()];
        let leaves1 = build_combined_layer(&f1, &cp1_dummy);
        let com1 = CombinedPoseidonCommitment::commit(&leaves1);

        let mut rngq = StdRng::seed_from_u64(333);
        for _ in 0..10 {
            let i = rngq.gen::<usize>() % n0;
            let leaf_i = leaves0[i];
            let proof_i = com0.open(i);
            let (neighbor_idx, neighbor_leaves, neighbor_proofs) = open_bucket_neighbors(&com0, &leaves0, i, m);
            let b = i / m;
            let parent_leaf = leaves1[b];
            let parent_proof = com1.open(b);
            let ok = verify_local_check_with_openings(
                &com0, &com1, i, leaf_i, &proof_i,
                &neighbor_idx, &neighbor_leaves, &neighbor_proofs,
                b, parent_leaf, &parent_proof,
                z0, m, omega0, n0,
            );
            assert!(ok, "local verification failed at i={}", i);
        }
    }

    #[test]
    fn test_fri_queries_roundtrip() {
        let n0 = 128usize;
        let schedule = vec![8usize, 8usize, 2usize];
        let r = 32usize;

        let dom = FriDomain::new_radix2(n0);
        let mut rng = StdRng::seed_from_u64(2025);
        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();

        let seed_z = 0x1234_5678_ABCDu64;
        let st = fri_build_transcript(f0, dom, &FriProverParams { schedule: schedule.clone(), seed_z });
        let roots: Vec<F> = st.transcript.layers.iter().map(|l| l.root).collect();
        let roots_seed = fs_seed_from_roots(&roots);
        let (queries, roots2) = fri_prove_queries(&st, r, roots_seed);
        assert_eq!(roots, roots2);

        let ok = fri_verify_queries(&schedule, n0, F::one(), seed_z, &roots, &queries, r);
        assert!(ok);
    }

    #[test]
    fn test_fri_queries_reject_corruption() {
        let n0 = 128usize;
        let schedule = vec![8usize, 8usize, 2usize];
        let r = 16usize;

        let dom = FriDomain::new_radix2(n0);
        let mut rng = StdRng::seed_from_u64(9090);
        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();

        let seed_z = 0xCAFE_F00D_u64;
        let st = fri_build_transcript(f0, dom, &FriProverParams { schedule: schedule.clone(), seed_z });
        let roots: Vec<F> = st.transcript.layers.iter().map(|l| l.root).collect();
        let roots_seed = fs_seed_from_roots(&roots);
        let (mut queries, _) = fri_prove_queries(&st, r, roots_seed);

        let q0 = 0usize; let ell0 = 0usize;
        if let Some(lo) = queries[q0].per_layer.get_mut(ell0) {
            if !lo.neighbor_leaves.is_empty() {
                lo.neighbor_leaves[0].f += F::one();
            }
        }

        let ok = fri_verify_queries(&schedule, n0, F::one(), seed_z, &roots, &queries, r);
        assert!(!ok);
    }
}