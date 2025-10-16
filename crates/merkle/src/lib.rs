use ark_pallas::Fr as F;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use poseidon::{
    hash_with_ds, hash_with_ds_dynamic, params::generate_params_t17_x5,
    poseidon_params_for_arity, poseidon_params_for_width, PoseidonParams, PoseidonParamsDynamic,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

// Wrapper that provides serde for field elements by encoding canonical bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerFr(pub F);

impl From<F> for SerFr {
    fn from(x: F) -> Self {
        SerFr(x)
    }
}
impl From<SerFr> for F {
    fn from(w: SerFr) -> F {
        w.0
    }
}

impl Serialize for SerFr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::new();
        self.0
            .serialize_with_mode(&mut bytes, Compress::Yes)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for SerFr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BytesVisitor;
        impl<'de> serde::de::Visitor<'de> for BytesVisitor {
            type Value = Vec<u8>;
            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "canonical ark-ff field bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(v.to_vec())
            }
            fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(v)
            }
        }
        let bytes = deserializer.deserialize_bytes(BytesVisitor)?;
        let f = F::deserialize_with_mode(&*bytes, Compress::Yes, Validate::Yes)
            .map_err(serde::de::Error::custom)?;
        Ok(SerFr(f))
    }
}

// Domain-separation label for Merkle nodes.
#[derive(Clone, Copy, Debug)]
pub struct DsLabel {
    pub arity: usize,
    pub level: u32,     // 0 = parents of leaves
    pub position: u64,  // node index at this level (or salt)
    pub tree_label: u64,
}

impl DsLabel {
    fn to_fields(self) -> [F; 4] {
        [
            F::from(self.arity as u64),
            F::from(self.level as u64),
            F::from(self.position),
            F::from(self.tree_label),
        ]
    }
}

// Updated: delegate arity→Poseidon width mapping to poseidon::poseidon_params_for_arity.
// Supports arities up to 64 (t ∈ {9, 17, 33, 65}).
fn params_for_arity(arity: usize) -> PoseidonParamsDynamic {
    poseidon_params_for_arity(arity)
}

// Parameterized Merkle configuration with explicit Poseidon params.
#[derive(Clone)]
pub struct MerkleChannelCfg {
    pub arity: usize,
    pub params: PoseidonParamsDynamic,
    pub tree_label: u64,
}

impl MerkleChannelCfg {
    pub fn with_params(arity: usize, params: PoseidonParamsDynamic) -> Self {
        Self {
            arity,
            params,
            tree_label: 0,
        }
    }

    pub fn new(arity: usize) -> Self {
        let params = params_for_arity(arity);
        Self {
            arity,
            params,
            tree_label: 0,
        }
    }

    pub fn with_tree_label(mut self, label: u64) -> Self {
        self.tree_label = label;
        self
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MerkleTree {
    pub leaves: Vec<SerFr>,
    pub root: SerFr,
    // Legacy field preserved for compatibility; not used by new DS hashing.
    pub ds_tag: SerFr,
    // level 0 = leaves (as digests), higher levels are parent digests
    pub levels: Vec<Vec<SerFr>>,
    // We skip serializing PoseidonParams; on deserialize, fill it via default_params().
    #[serde(skip, default = "default_params")]
    pub params: PoseidonParams,
    // New dynamic params are not serialized; derive from arity where needed.
    #[serde(skip)]
    pub cfg: Option<MerkleChannelCfg>,
}

// Union-of-paths multiproof (single representation used by both single-column and pair-leaf trees).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MerkleProof {
    // Requested leaf indices in ascending order (unique).
    pub indices: Vec<usize>,
    // For each level ℓ, siblings[ℓ] is a flat list of sibling node digests needed
    // to complete all parents touched at that level (union-of-paths).
    pub siblings: Vec<Vec<SerFr>>,
    // For each level ℓ, group_sizes[ℓ] lists, in order, the child_count for each touched parent.
    // This drives reconstruction deterministically.
    pub group_sizes: Vec<Vec<u8>>,
    // Arity of the tree.
    pub arity: usize,
}

impl MerkleTree {
    // ========== Single-column DS-aware constructor ==========
    pub fn new(leaves: Vec<F>, cfg: MerkleChannelCfg) -> Self {
        assert!(!leaves.is_empty(), "no leaves");
        let arity = cfg.arity;

        let mut levels: Vec<Vec<F>> = Vec::new();
        levels.push(leaves);

        // Extended width checks: arity bucket must match t ∈ {9, 17, 33, 65}
        let t = cfg.params.t;
        let ok_width = (arity <= 8 && t == 9)
            || (arity >= 9 && arity <= 16 && t == 17)
            || (arity >= 17 && arity <= 32 && t == 33)
            || (arity >= 33 && arity <= 64 && t == 65);
        assert!(ok_width, "arity {} incompatible with Poseidon width t={}", arity, t);

        let mut cur_level = 0u32;
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity((cur.len() + arity - 1) / arity);
            for (parent_idx, chunk) in cur.chunks(arity).enumerate() {
                let ds = DsLabel {
                    arity,
                    level: cur_level,
                    position: parent_idx as u64,
                    tree_label: cfg.tree_label,
                };
                let digest = hash_with_ds_dynamic(&ds.to_fields(), chunk, &cfg.params);
                next.push(digest);
            }
            levels.push(next);
            cur_level += 1;
        }
        let root = *levels.last().unwrap().first().unwrap();

        MerkleTree {
            leaves: levels[0].iter().copied().map(SerFr::from).collect(),
            root: SerFr(root),
            ds_tag: SerFr(F::from(0u64)),
            levels: levels
                .into_iter()
                .map(|v| v.into_iter().map(SerFr::from).collect())
                .collect(),
            params: default_params(),
            cfg: Some(cfg),
        }
    }

    // Legacy API preserved: uses fixed t=17 hashing with a single ds_tag in capacity.
    pub fn new_legacy(leaves: Vec<F>, ds_tag: F, params: PoseidonParams) -> Self {
        assert!(!leaves.is_empty(), "no leaves");

        let mut levels: Vec<Vec<F>> = Vec::new();
        levels.push(leaves);
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity((cur.len() + poseidon::RATE - 1) / poseidon::RATE);
            for chunk in cur.chunks(poseidon::RATE) {
                let digest = hash_with_ds(chunk, ds_tag, &params);
                next.push(digest);
            }
            levels.push(next);
        }
        let root = *levels.last().unwrap().first().unwrap();

        MerkleTree {
            leaves: levels[0].iter().copied().map(SerFr::from).collect(),
            root: SerFr(root),
            ds_tag: SerFr(ds_tag),
            levels: levels
                .into_iter()
                .map(|v| v.into_iter().map(SerFr::from).collect())
                .collect(),
            params,
            cfg: None,
        }
    }

    pub fn root(&self) -> F {
        self.root.0
    }

    pub fn arity(&self) -> usize {
        if let Some(cfg) = &self.cfg {
            cfg.arity
        } else {
            poseidon::RATE
        }
    }

    pub fn height(&self) -> usize {
        if self.levels.is_empty() {
            0
        } else {
            self.levels.len() - 1
        }
    }

    // ========== Union-of-paths encoder used by both single and pair paths ==========
    fn open_union_of_paths(&self, indices: &[usize]) -> MerkleProof {
        assert!(!indices.is_empty(), "open_many: empty indices");
        let arity = self.arity();

        let leaf_count = self.levels[0].len();
        debug_assert!(indices.iter().all(|&i| i < leaf_count));

        // Work on sorted unique indices
        let mut cur_indices: Vec<usize> = indices.to_vec();
        cur_indices.sort_unstable();
        cur_indices.dedup();

        let mut siblings_per_level: Vec<Vec<SerFr>> = Vec::with_capacity(self.height());
        let mut group_sizes_per_level: Vec<Vec<u8>> = Vec::with_capacity(self.height());

        for level in 0..self.height() {
            let level_nodes = &self.levels[level];
            let level_len = level_nodes.len();

            use std::collections::BTreeMap;
            let mut map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for &i in &cur_indices {
                let p = i / arity;
                let cpos = i % arity;
                map.entry(p).or_default().push(cpos);
            }

            let mut level_siblings: Vec<SerFr> = Vec::new();
            let mut level_group_sizes: Vec<u8> = Vec::new();

            for (parent_idx, mut opened_positions) in map {
                opened_positions.sort_unstable();

                let base = parent_idx * arity;
                let end = core::cmp::min(base + arity, level_len);
                let child_count = end - base;
                debug_assert!((1..=arity).contains(&child_count));
                level_group_sizes.push(child_count as u8);

                let mut opened_iter = opened_positions.iter().copied().peekable();
                for child_pos in 0..child_count {
                    if opened_iter.peek().copied() == Some(child_pos) {
                        opened_iter.next();
                    } else {
                        level_siblings.push(level_nodes[base + child_pos]);
                    }
                }
            }

            siblings_per_level.push(level_siblings);
            group_sizes_per_level.push(level_group_sizes);

            let mut next_indices: Vec<usize> = cur_indices.iter().map(|&i| i / arity).collect();
            next_indices.sort_unstable();
            next_indices.dedup();
            cur_indices = next_indices;
        }

        MerkleProof {
            indices: {
                let mut idx = indices.to_vec();
                idx.sort_unstable();
                idx.dedup();
                idx
            },
            siblings: siblings_per_level,
            group_sizes: group_sizes_per_level,
            arity,
        }
    }

    // ========== Single-column: open many (multiproof) ==========
    pub fn open_many_single(&self, indices: &[usize]) -> MerkleProof {
        self.open_union_of_paths(indices)
    }

    // Existing multiproof (used by legacy and pairs). Kept for compatibility.
    pub fn open_many(&self, indices: &[usize]) -> MerkleProof {
        self.open_union_of_paths(indices)
    }

    // Debug-only consistency checker: recompute level parents and compare.
    fn check_level_consistency(&self, level: usize) -> bool {
        let arity = self.arity();
        if level >= self.height() {
            return true;
        }
        let cur = &self.levels[level];
        let next = &self.levels[level + 1];

        let expected_parents = (cur.len() + arity - 1) / arity;
        if next.len() != expected_parents {
            return false;
        }
        for parent_idx in 0..expected_parents {
            let base = parent_idx * arity;
            let end = core::cmp::min(base + arity, cur.len());
            let children: Vec<F> = cur[base..end].iter().map(|w| w.0).collect();

            let digest = if let Some(cfg) = &self.cfg {
                let ds = DsLabel {
                    arity,
                    level: level as u32,
                    position: parent_idx as u64,
                    tree_label: cfg.tree_label,
                };
                hash_with_ds_dynamic(&ds.to_fields(), &children, &cfg.params)
            } else {
                hash_with_ds(&children, self.ds_tag.0, &self.params)
            };

            if digest != next[parent_idx].0 {
                return false;
            }
        }
        true
    }
}

// Legacy default params (t=17).
pub fn default_params() -> PoseidonParams {
    let seed = b"POSEIDON-T17-X5-SEED";
    generate_params_t17_x5(seed)
}

// ========== Combined-leaf hashing (pack (f, cp) into a single absorb) ==========

fn encode_leaf_digest_legacy(f: F, cp: F, ds_tag: F, params: &PoseidonParams) -> F {
    hash_with_ds(&[f, cp], ds_tag, params)
}

// For DS-aware encoding, dedicate a special level marker for leaves.
const LEAF_LEVEL_DS: u32 = u32::MAX;

fn encode_leaf_digest_ds(index: usize, cfg: &MerkleChannelCfg, f: F, cp: F) -> F {
    let ds = DsLabel {
        arity: cfg.arity,
        level: LEAF_LEVEL_DS,
        position: index as u64,
        tree_label: cfg.tree_label,
    };
    hash_with_ds_dynamic(&ds.to_fields(), &[f, cp], &cfg.params)
}

impl MerkleTree {
    // Build a Merkle tree from pairs (f, cp) using DS-aware leaf encoding and internal DS-aware nodes.
    pub fn new_pairs(f_vals: &[F], cp_vals: &[F], cfg: MerkleChannelCfg) -> Self {
        assert_eq!(f_vals.len(), cp_vals.len(), "f and cp length mismatch");
        assert!(!f_vals.is_empty(), "no leaves");
        let n = f_vals.len();

        let mut level0: Vec<F> = Vec::with_capacity(n);
        for i in 0..n {
            level0.push(encode_leaf_digest_ds(i, &cfg, f_vals[i], cp_vals[i]));
        }

        let arity = cfg.arity;
        let mut levels: Vec<Vec<F>> = Vec::new();
        levels.push(level0);

        // Extended width checks for pairs path
        let t = cfg.params.t;
        let ok_width = (arity <= 8 && t == 9)
            || (arity >= 9 && arity <= 16 && t == 17)
            || (arity >= 17 && arity <= 32 && t == 33)
            || (arity >= 33 && arity <= 64 && t == 65);
        assert!(ok_width, "arity {} incompatible with Poseidon width t={}", arity, t);

        let mut cur_level = 0u32; // 0 = parents of leaves
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity((cur.len() + arity - 1) / arity);
            for (parent_idx, chunk) in cur.chunks(arity).enumerate() {
                let ds = DsLabel {
                    arity,
                    level: cur_level,
                    position: parent_idx as u64,
                    tree_label: cfg.tree_label,
                };
                let digest = hash_with_ds_dynamic(&ds.to_fields(), chunk, &cfg.params);
                next.push(digest);
            }
            levels.push(next);
            cur_level += 1;
        }
        let root = *levels.last().unwrap().first().unwrap();

        MerkleTree {
            leaves: levels[0].iter().copied().map(SerFr::from).collect(),
            root: SerFr(root),
            ds_tag: SerFr(F::from(0u64)), // unused in DS path
            levels: levels
                .into_iter()
                .map(|v| v.into_iter().map(SerFr::from).collect())
                .collect(),
            params: default_params(), // legacy fixed params unused here
            cfg: Some(cfg),
        }
    }

    // Legacy combined-leaf constructor
    pub fn new_pairs_legacy(f_vals: &[F], cp_vals: &[F], ds_tag: F, params: PoseidonParams) -> Self {
        assert_eq!(f_vals.len(), cp_vals.len(), "f and cp length mismatch");
        assert!(!f_vals.is_empty(), "no leaves");
        let n = f_vals.len();

        let mut level0: Vec<F> = Vec::with_capacity(n);
        for i in 0..n {
            let d = encode_leaf_digest_legacy(f_vals[i], cp_vals[i], ds_tag, &params);
            level0.push(d);
        }

        let mut levels: Vec<Vec<F>> = Vec::new();
        levels.push(level0);
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity((cur.len() + poseidon::RATE - 1) / poseidon::RATE);
            for chunk in cur.chunks(poseidon::RATE) {
                let digest = hash_with_ds(chunk, ds_tag, &params);
                next.push(digest);
            }
            levels.push(next);
        }
        let root = *levels.last().unwrap().first().unwrap();

        MerkleTree {
            leaves: levels[0].iter().copied().map(SerFr::from).collect(),
            root: SerFr(root),
            ds_tag: SerFr(ds_tag),
            levels: levels
                .into_iter()
                .map(|v| v.into_iter().map(SerFr::from).collect())
                .collect(),
            params,
            cfg: None,
        }
    }
}

// ========== Legacy verifications (unchanged behavior) ==========
pub fn verify_many(
    root: &F,
    indices: &[usize],
    values: &[F],
    proof: &MerkleProof,
    ds_tag: F,
    params: PoseidonParams,
) -> bool {
    if indices.is_empty() || indices.len() != values.len() {
        return false;
    }
    // We accept indices in any order from the caller, but proof.indices is unique-sorted.
    let mut req = indices.to_vec();
    req.sort_unstable();
    req.dedup();
    if proof.indices != req {
        return false;
    }
    if proof.siblings.len() != proof.group_sizes.len() {
        return false;
    }
    let arity = proof.arity;

    // Prepare current frontier exactly over the requested (unique-sorted) set.
    let mut cur_indices = req;
    // Map the original indices -> value; then assemble leaves aligned to cur_indices order.
    use std::collections::BTreeMap;
    let mut map: BTreeMap<usize, F> = BTreeMap::new();
    for (&i, &v) in indices.iter().zip(values.iter()) {
        map.insert(i, v);
    }
    let mut cur_values: Vec<F> = cur_indices.iter().map(|i| map[i]).collect();

    for (level_siblings, level_group_sizes) in proof.siblings.iter().zip(proof.group_sizes.iter())
    {
        let mut groups: BTreeMap<usize, Vec<(usize, F)>> = BTreeMap::new();
        for (idx, val) in cur_indices.iter().copied().zip(cur_values.iter().copied()) {
            let p = idx / arity;
            let cpos = idx % arity;
            groups.entry(p).or_default().push((cpos, val));
        }

        if groups.len() != level_group_sizes.len() {
            return false;
        }

        let mut next_indices: Vec<usize> = Vec::with_capacity(groups.len());
        let mut next_values: Vec<F> = Vec::with_capacity(groups.len());

        let mut off = 0usize;

        for ((parent_idx, mut opened), child_count_u8) in
            groups.into_iter().zip(level_group_sizes.iter().copied())
        {
            let child_count = child_count_u8 as usize;
            if child_count == 0 || child_count > arity {
                return false;
            }

            opened.sort_unstable_by_key(|(cpos, _)| *cpos);

            let mut opened_iter = opened.into_iter().peekable();
            let mut children: Vec<F> = Vec::with_capacity(child_count);

            for child_pos in 0..child_count {
                if let Some(&(cpos, val)) = opened_iter.peek() {
                    if cpos == child_pos {
                        children.push(val);
                        opened_iter.next();
                        continue;
                    }
                }
                if off >= level_siblings.len() {
                    return false;
                }
                children.push(level_siblings[off].0);
                off += 1;
            }

            let parent = hash_with_ds(&children, ds_tag, &params);

            next_indices.push(parent_idx);
            next_values.push(parent);
        }

        if off != level_siblings.len() {
            return false;
        }

        cur_indices = next_indices;
        cur_values = next_values;
    }

    if cur_values.len() != 1 {
        return false;
    }
    cur_values[0] == *root
}

// New DS-hygienic verification API (explicit) for single-column values.
pub fn verify_many_ds(
    root: &F,
    indices: &[usize],
    values: &[F],
    proof: &MerkleProof,
    tree_label: u64,
    dyn_params: PoseidonParamsDynamic,
) -> bool {
    if indices.is_empty() || indices.len() != values.len() {
        return false;
    }
    let mut req = indices.to_vec();
    req.sort_unstable();
    req.dedup();
    if proof.indices != req {
        return false;
    }
    if proof.siblings.len() != proof.group_sizes.len() {
        return false;
    }
    let arity = proof.arity;

    // Extended width guard
    let t = dyn_params.t;
    let ok_width = (arity <= 8 && t == 9)
        || (arity >= 9 && arity <= 16 && t == 17)
        || (arity >= 17 && arity <= 32 && t == 33)
        || (arity >= 33 && arity <= 64 && t == 65);
    if !ok_width {
        return false;
    }

    // Align leaves to proof.indices order.
    use std::collections::BTreeMap;
    let mut map: BTreeMap<usize, F> = BTreeMap::new();
    for (&i, &v) in indices.iter().zip(values.iter()) {
        map.insert(i, v);
    }
    let mut cur_indices = req;
    let mut cur_values: Vec<F> = cur_indices.iter().map(|i| map[i]).collect();

    for (level, (level_siblings, level_group_sizes)) in
        proof.siblings.iter().zip(proof.group_sizes.iter()).enumerate()
    {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<usize, Vec<(usize, F)>> = BTreeMap::new();
        for (idx, val) in cur_indices.iter().copied().zip(cur_values.iter().copied()) {
            let p = idx / arity;
            let cpos = idx % arity;
            groups.entry(p).or_default().push((cpos, val));
        }

        if groups.len() != level_group_sizes.len() {
            return false;
        }

        let mut next_indices: Vec<usize> = Vec::with_capacity(groups.len());
        let mut next_values: Vec<F> = Vec::with_capacity(groups.len());

        let mut off = 0usize;

        for ((parent_idx, mut opened), child_count_u8) in
            groups.into_iter().zip(level_group_sizes.iter().copied())
        {
            let child_count = child_count_u8 as usize;
            if child_count == 0 || child_count > arity {
                return false;
            }

            opened.sort_unstable_by_key(|(cpos, _)| *cpos);

            let mut opened_iter = opened.into_iter().peekable();
            let mut children: Vec<F> = Vec::with_capacity(child_count);

            for child_pos in 0..child_count {
                if let Some(&(cpos, val)) = opened_iter.peek() {
                    if cpos == child_pos {
                        children.push(val);
                        opened_iter.next();
                        continue;
                    }
                }
                if off >= level_siblings.len() {
                    return false;
                }
                children.push(level_siblings[off].0);
                off += 1;
            }

            let ds = DsLabel {
                arity,
                level: level as u32,
                position: parent_idx as u64,
                tree_label,
            };
            let parent = hash_with_ds_dynamic(&ds.to_fields(), &children, &dyn_params);

            next_indices.push(parent_idx);
            next_values.push(parent);
        }

        if off != level_siblings.len() {
            return false;
        }

        cur_indices = next_indices;
        cur_values = next_values;
    }

    if cur_values.len() != 1 {
        return false;
    }
    cur_values[0] == *root
}

// Verify pairs under legacy mode: recompute leaf digests from (f,cp) pairs and then verify path.
pub fn verify_pairs_legacy(
    root: &F,
    indices: &[usize],
    pairs: &[(F, F)],
    proof: &MerkleProof,
    ds_tag: F,
    params: PoseidonParams,
) -> bool {
    if indices.len() != pairs.len() || indices.is_empty() {
        return false;
    }
    let leaves: Vec<F> = pairs
        .iter()
        .map(|&(f, cp)| encode_leaf_digest_legacy(f, cp, ds_tag, &params))
        .collect();
    verify_many(root, indices, &leaves, proof, ds_tag, params)
}

// Verify pairs under DS-aware mode: recompute leaf digests with leaf DS and then verify with DS-aware internal hashing.
pub fn verify_pairs_ds(
    root: &F,
    indices: &[usize],
    pairs: &[(F, F)],
    proof: &MerkleProof,
    tree_label: u64,
    dyn_params: PoseidonParamsDynamic,
) -> bool {
    if indices.len() != pairs.len() || indices.is_empty() {
        return false;
    }
    let arity = proof.arity;

    // Extended width guard
    let t = dyn_params.t;
    let ok_width = (arity <= 8 && t == 9)
        || (arity >= 9 && arity <= 16 && t == 17)
        || (arity >= 17 && arity <= 32 && t == 33)
        || (arity >= 33 && arity <= 64 && t == 65);
    if !ok_width {
        return false;
    }

    // Recompute leaf digests using DS policy (LEAF_LEVEL_DS).
    // Align leaves to proof.indices order to match union-of-paths verifier expectations.
    let mut req = indices.to_vec();
    req.sort_unstable();
    req.dedup();

    use std::collections::BTreeMap;
    let mut mpairs: BTreeMap<usize, (F, F)> = BTreeMap::new();
    for (&i, &p) in indices.iter().zip(pairs.iter()) {
        mpairs.insert(i, p);
    }
    let leaves: Vec<F> = req
        .iter()
        .map(|&idx| {
            let (f, cp) = mpairs[&idx];
            let ds = DsLabel {
                arity,
                level: LEAF_LEVEL_DS,
                position: idx as u64,
                tree_label,
            };
            hash_with_ds_dynamic(&ds.to_fields(), &[f, cp], &dyn_params)
        })
        .collect();

    verify_many_ds(root, &req, &leaves, proof, tree_label, dyn_params)
}

// ========== Small facades for ergonomics ==========

pub struct MerkleProver {
    pub cfg: MerkleChannelCfg,
}

impl MerkleProver {
    pub fn new(cfg: MerkleChannelCfg) -> Self {
        Self { cfg }
    }

    // Commit a vector of single-column leaves (already digests or raw values you wish to commit).
    pub fn commit_single(&self, leaves: &[F]) -> (F, MerkleTree) {
        let tree = MerkleTree::new(leaves.to_vec(), self.cfg.clone());
        (tree.root(), tree)
    }

    // Open single-column leaves at given indices (union-of-paths multiproof).
    pub fn open_single(&self, tree: &MerkleTree, indices: &[usize]) -> MerkleProof {
        tree.open_many_single(indices)
    }

    // Verify single-column union-of-paths proof with DS-aware hashing.
    pub fn verify_single(
        &self,
        root: &F,
        indices: &[usize],
        leaves: &[F],
        proof: &MerkleProof,
    ) -> bool {
        verify_many_ds(
            root,
            indices,
            leaves,
            proof,
            self.cfg.tree_label,
            self.cfg.params.clone(),
        )
    }

    // Commit a vector of pairs (f, cp) as combined leaves; returns root and the constructed tree.
    pub fn commit_pairs(&self, f_vals: &[F], cp_vals: &[F]) -> (F, MerkleTree) {
        let tree = MerkleTree::new_pairs(f_vals, cp_vals, self.cfg.clone());
        (tree.root(), tree)
    }

    // Open a set of indices; returns the original pairs at those indices and the Merkle proof.
    pub fn open_pairs(
        &self,
        tree: &MerkleTree,
        f_vals: &[F],
        cp_vals: &[F],
        indices: &[usize],
    ) -> (Vec<(F, F)>, MerkleProof) {
        assert_eq!(f_vals.len(), cp_vals.len(), "length mismatch");
        assert!(!indices.is_empty(), "empty indices");
        let mut uniq = indices.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let pairs: Vec<(F, F)> = uniq.iter().map(|&i| (f_vals[i], cp_vals[i])).collect();
        let proof = tree.open_many(&uniq);
        (pairs, proof)
    }

    pub fn verify_pairs(
        &self,
        root: &F,
        indices: &[usize],
        pairs: &[(F, F)],
        proof: &MerkleProof,
    ) -> bool {
        verify_pairs_ds(
            root,
            indices,
            pairs,
            proof,
            self.cfg.tree_label,
            self.cfg.params.clone(),
        )
    }
}

pub struct LegacyMerkleProver {
    pub ds_tag: F,
    pub params: PoseidonParams,
}

impl LegacyMerkleProver {
    pub fn new(ds_tag: F, params: PoseidonParams) -> Self {
        Self { ds_tag, params }
    }

    pub fn commit_pairs(&self, f_vals: &[F], cp_vals: &[F]) -> (F, MerkleTree) {
        let tree = MerkleTree::new_pairs_legacy(f_vals, cp_vals, self.ds_tag, self.params.clone());
        (tree.root(), tree)
    }

    pub fn open_pairs(
        &self,
        tree: &MerkleTree,
        f_vals: &[F],
        cp_vals: &[F],
        indices: &[usize],
    ) -> (Vec<(F, F)>, MerkleProof) {
        assert_eq!(f_vals.len(), cp_vals.len(), "length mismatch");
        assert!(!indices.is_empty(), "empty indices");
        let mut uniq = indices.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let pairs: Vec<(F, F)> = uniq.iter().map(|&i| (f_vals[i], cp_vals[i])).collect();
        let proof = tree.open_many(&uniq);
        (pairs, proof)
    }

    pub fn verify_pairs(
        &self,
        root: &F,
        indices: &[usize],
        pairs: &[(F, F)],
        proof: &MerkleProof,
    ) -> bool {
        verify_pairs_legacy(
            root,
            indices,
            pairs,
            proof,
            self.ds_tag,
            self.params.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{UniformRand, Zero};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn merkle_proof_roundtrip_arbitrary_size_legacy() {
        let mut rng = StdRng::seed_from_u64(123);
        let n = 55usize;
        let leaves: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let params = default_params();
        let ds = F::from(77u64);
        let tree = MerkleTree::new_legacy(leaves.clone(), ds, params.clone());

        assert!(tree.check_level_consistency(0));

        let root = tree.root();
        let mut idx = vec![0usize, 3, 7, 11, 54];
        idx.sort_unstable();
        idx.dedup();
        let vals: Vec<F> = idx.iter().map(|&i| leaves[i]).collect();
        let proof = tree.open_many(&idx);
        assert!(verify_many(&root, &idx, &vals, &proof, ds, params));
    }

    #[test]
    fn merkle_roundtrip_arity16_ds_hygiene() {
        let mut rng = StdRng::seed_from_u64(999);
        let n = 64usize;
        let leaves: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let cfg = MerkleChannelCfg::new(16).with_tree_label(42);
        let tree = MerkleTree::new(leaves.clone(), cfg.clone());

        assert!(tree.check_level_consistency(0));
        if tree.height() >= 2 {
            assert!(tree.check_level_consistency(1));
        }

        let root = tree.root();
        let mut idx = vec![0usize, 15, 16, 31, 47, 63];
        idx.sort_unstable();
        idx.dedup();
        let vals: Vec<F> = idx.iter().map(|&i| leaves[i]).collect();
        let proof = tree.open_many_single(&idx);

        let dyn_params = poseidon_params_for_width(16 + 1);
        assert!(verify_many_ds(
            &root,
            &idx,
            &vals,
            &proof,
            cfg.tree_label,
            dyn_params
        ));
    }

    #[test]
    fn test_poseidon_params_roundtrip_t17() {
        let params = poseidon_params_for_width(17);

        let children: Vec<F> = (0..16).map(|i| F::from(i as u64 + 1)).collect();
        let arity = 16usize;
        let level = 0u32;
        let position = 3u64;
        let tree_label = 42u64;

        let ds = DsLabel {
            arity,
            level,
            position,
            tree_label,
        };
        let digest1 = hash_with_ds_dynamic(&ds.to_fields(), &children, &params);
        let digest2 = hash_with_ds_dynamic(&ds.to_fields(), &children, &params);
        assert_eq!(digest1, digest2);

        let ds_level = DsLabel { level: level + 1, ..ds };
        let d_level = hash_with_ds_dynamic(&ds_level.to_fields(), &children, &params);
        assert_ne!(digest1, d_level);

        let ds_pos = DsLabel { position: position + 1, ..ds };
        let d_pos = hash_with_ds_dynamic(&ds_pos.to_fields(), &children, &params);
        assert_ne!(digest1, d_pos);

        let ds_tree = DsLabel { tree_label: tree_label + 1, ..ds };
        let d_tree = hash_with_ds_dynamic(&ds_tree.to_fields(), &children, &params);
        assert_ne!(digest1, d_tree);

        let ds_arity8 = DsLabel { arity: 8, ..ds };
        let d_arity8 = hash_with_ds_dynamic(&ds_arity8.to_fields(), &children, &params);
        assert_ne!(digest1, d_arity8);

        let fewer_children: Vec<F> = (0..5).map(|i| F::from(i as u64 + 1)).collect();
        let digest_few_1 = hash_with_ds_dynamic(&ds.to_fields(), &fewer_children, &params);
        let digest_few_2 = hash_with_ds_dynamic(&ds.to_fields(), &fewer_children, &params);
        assert_eq!(digest_few_1, digest_few_2);

        let mut with_extra_zero = fewer_children.clone();
        with_extra_zero.push(F::zero());
        let digest_with_extra = hash_with_ds_dynamic(&ds.to_fields(), &with_extra_zero, &params);
        assert_ne!(digest_few_1, digest_with_extra);
    }

    #[test]
    fn test_poseidon_params_roundtrip_t9() {
        let params = poseidon_params_for_width(9);

        let children: Vec<F> = (0..8).map(|i| F::from(i as u64 + 11)).collect();
        let arity = 8usize;
        let level = 2u32;
        let position = 5u64;
        let tree_label = 7u64;

        let ds = DsLabel {
            arity,
            level,
            position,
            tree_label,
        };
        let digest1 = hash_with_ds_dynamic(&ds.to_fields(), &children, &params);
        let digest2 = hash_with_ds_dynamic(&ds.to_fields(), &children, &params);
        assert_eq!(digest1, digest2);

        let d_level = hash_with_ds_dynamic(&DsLabel { level: level + 1, ..ds }.to_fields(), &children, &params);
        assert_ne!(digest1, d_level);

        let d_pos = hash_with_ds_dynamic(&DsLabel { position: position + 1, ..ds }.to_fields(), &children, &params);
        assert_ne!(digest1, d_pos);

        let d_tree = hash_with_ds_dynamic(&DsLabel { tree_label: tree_label + 1, ..ds }.to_fields(), &children, &params);
        assert_ne!(digest1, d_tree);

        let d_arity16 = hash_with_ds_dynamic(&DsLabel { arity: 16, ..ds }.to_fields(), &children, &params);
        assert_ne!(digest1, d_arity16);

        let fewer_children: Vec<F> = (0..3).map(|i| F::from(i as u64 + 21)).collect();
        let digest_few = hash_with_ds_dynamic(&ds.to_fields(), &fewer_children, &params);
        let mut with_extra_zero = fewer_children.clone();
        with_extra_zero.push(F::zero());
        let digest_extra = hash_with_ds_dynamic(&ds.to_fields(), &with_extra_zero, &params);
        assert_ne!(digest_few, digest_extra);
    }

    #[test]
    fn merkle_ds_hygiene_negatives_arity16() {
        let leaves: Vec<F> = (1..=32).map(|x| F::from(x as u64)).collect();
        let cfg = MerkleChannelCfg::new(16).with_tree_label(1234);
        let tree = MerkleTree::new(leaves.clone(), cfg.clone());

        assert!(tree.check_level_consistency(0));

        let arity = cfg.arity;
        let level0 = 0u32;
        let parent_idx = 1usize;
        let base = parent_idx * arity;
        let end = core::cmp::min(base + arity, tree.levels[0].len());
        let children: Vec<F> = tree.levels[0][base..end].iter().map(|w| w.0).collect();

        let ds = DsLabel { arity, level: level0, position: parent_idx as u64, tree_label: cfg.tree_label };
        let parent_digest = hash_with_ds_dynamic(&ds.to_fields(), &children, &cfg.params);
        assert_eq!(parent_digest, tree.levels[1][parent_idx].0);

        let d2 = hash_with_ds_dynamic(&DsLabel { level: level0 + 1, ..ds }.to_fields(), &children, &cfg.params);
        assert_ne!(parent_digest, d2);

        let d3 = hash_with_ds_dynamic(&DsLabel { position: (parent_idx as u64) + 1, ..ds }.to_fields(), &children, &cfg.params);
        assert_ne!(parent_digest, d3);

        let d4 = hash_with_ds_dynamic(&DsLabel { tree_label: cfg.tree_label + 1, ..ds }.to_fields(), &children, &cfg.params);
        assert_ne!(parent_digest, d4);

        let mut shuffled = children.clone();
        if shuffled.len() >= 2 { shuffled.swap(0, 1); }
        let d5 = hash_with_ds_dynamic(&ds.to_fields(), &shuffled, &cfg.params);
        assert_ne!(parent_digest, d5);
    }

    #[test]
    fn test_combined_leaf_commit_open_legacy() {
        let mut rng = StdRng::seed_from_u64(2024);
        let n = 37usize;
        let f_vals: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let cp_vals: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let params = default_params();
        let ds_tag = F::from(99u64);
        let tree = MerkleTree::new_pairs_legacy(&f_vals, &cp_vals, ds_tag, params.clone());
        let root = tree.root();

        let mut idx = vec![0usize, 1, 5, 19, 36];
        idx.sort_unstable();
        idx.dedup();
        let pairs: Vec<(F, F)> = idx.iter().map(|&i| (f_vals[i], cp_vals[i])).collect();

        let proof = tree.open_many(&idx);
        assert!(verify_pairs_legacy(&root, &idx, &pairs, &proof, ds_tag, params));
    }

    #[test]
    fn test_combined_leaf_commit_open_ds_arity16() {
        let mut rng = StdRng::seed_from_u64(2025);
        let n = 64usize;
        let f_vals: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let cp_vals: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let cfg = MerkleChannelCfg::new(16).with_tree_label(777);
        let tree = MerkleTree::new_pairs(&f_vals, &cp_vals, cfg.clone());
        let root = tree.root();

        let mut idx = vec![0usize, 7, 16, 31, 63];
        idx.sort_unstable();
        idx.dedup();
        let pairs: Vec<(F, F)> = idx.iter().map(|&i| (f_vals[i], cp_vals[i])).collect();
        let proof = tree.open_many(&idx);

        let dyn_params = poseidon_params_for_width(16 + 1);
        assert!(verify_pairs_ds(&root, &idx, &pairs, &proof, cfg.tree_label, dyn_params));

        let mut tampered = pairs.clone();
        tampered[0].1 += F::from(1u64);
        assert!(!verify_pairs_ds(
            &root,
            &idx,
            &tampered,
            &proof,
            cfg.tree_label,
            poseidon_params_for_width(17)
        ));
    }

    #[test]
    fn test_combined_leaf_commit_open_ds_arity8() {
        let mut rng = StdRng::seed_from_u64(3030);
        let n = 32usize;
        let f_vals: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let cp_vals: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let cfg = MerkleChannelCfg::new(8).with_tree_label(8888);
        let tree = MerkleTree::new_pairs(&f_vals, &cp_vals, cfg.clone());
        let root = tree.root();

        let mut idx = vec![0usize, 3, 7, 8, 15, 23, 31];
        idx.sort_unstable();
        idx.dedup();
        let pairs: Vec<(F, F)> = idx.iter().map(|&i| (f_vals[i], cp_vals[i])).collect();
        let proof = tree.open_many(&idx);

        let dyn_params = poseidon_params_for_width(8 + 1);
        assert!(verify_pairs_ds(&root, &idx, &pairs, &proof, cfg.tree_label, dyn_params));

        let mut tampered = pairs.clone();
        tampered[2].0 += F::from(1u64);
        assert!(!verify_pairs_ds(
            &root,
            &idx,
            &tampered,
            &proof,
            cfg.tree_label,
            poseidon_params_for_width(9)
        ));

        // Prover facade smoke test (single and pairs)
        let prover = MerkleProver::new(cfg.clone());
        let (root2, tree2) = prover.commit_pairs(&f_vals, &cp_vals);
        assert_eq!(root, root2);
        let (pairs2, proof2) = prover.open_pairs(&tree2, &f_vals, &cp_vals, &idx);
        assert_eq!(pairs, pairs2);
        assert!(prover.verify_pairs(&root2, &idx, &pairs2, &proof2));

        // Single-column smoke test
        let (root3, tree3) = prover.commit_single(&f_vals);
        assert_eq!(root3, tree3.root());
        let proof3 = prover.open_single(&tree3, &idx);
        assert!(prover.verify_single(&root3, &idx, &idx.iter().map(|&i| f_vals[i]).collect::<Vec<_>>(), &proof3));
    }
}