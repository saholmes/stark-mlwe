use ark_pallas::Fr as F;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use poseidon::{
    hash_with_ds, hash_with_ds_dynamic, params::generate_params_t17_x5, poseidon_params_for_width,
    PoseidonParams, PoseidonParamsDynamic,
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
        let params = poseidon_params_for_width(arity + 1);
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

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MerkleProof {
    pub indices: Vec<usize>,
    pub siblings: Vec<Vec<SerFr>>,
    pub group_sizes: Vec<Vec<u8>>,
    pub arity: usize,
}

impl MerkleTree {
    // New API: explicit cfg with dynamic params and DS hygiene.
    pub fn new(leaves: Vec<F>, cfg: MerkleChannelCfg) -> Self {
        assert!(!leaves.is_empty(), "no leaves");
        let arity = cfg.arity;

        let mut levels: Vec<Vec<F>> = Vec::new();
        levels.push(leaves);

        let t = cfg.params.t;
        assert_eq!(t, arity + 1, "poseidon width t must equal arity + 1");

        // Build levels using DS labels: level index starts at 0 for parents of leaves.
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
            // Legacy ds_tag kept for serde compatibility; zero by default under new API.
            ds_tag: SerFr(F::from(0u64)),
            levels: levels
                .into_iter()
                .map(|v| v.into_iter().map(SerFr::from).collect())
                .collect(),
            // Legacy fixed params retained; used only by legacy verify/new_legacy.
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

    pub fn open_many(&self, indices: &[usize]) -> MerkleProof {
        assert!(!indices.is_empty(), "open_many: empty indices");
        let arity = self.arity();

        let leaf_count = self.levels[0].len();
        debug_assert!(indices.iter().all(|&i| i < leaf_count));

        let mut cur_indices: Vec<usize> = indices.to_vec();

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
            indices: indices.to_vec(),
            siblings: siblings_per_level,
            group_sizes: group_sizes_per_level,
            arity,
        }
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

pub fn verify_many(
    root: &F,
    indices: &[usize],
    values: &[F],
    proof: &MerkleProof,
    ds_tag: F,
    params: PoseidonParams,
) -> bool {
    // Legacy-only verification (matches original API and tests).
    if indices.is_empty() || indices.len() != values.len() {
        return false;
    }
    if proof.indices != indices {
        return false;
    }
    if proof.siblings.len() != proof.group_sizes.len() {
        return false;
    }
    let arity = proof.arity;

    let mut cur_indices = indices.to_vec();
    let mut cur_values = values.to_vec();

    for (level_siblings, level_group_sizes) in proof.siblings.iter().zip(proof.group_sizes.iter()) {
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

// New DS-hygienic verification API (explicit).
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
    if proof.indices != indices {
        return false;
    }
    if proof.siblings.len() != proof.group_sizes.len() {
        return false;
    }
    let arity = proof.arity;
    if dyn_params.t != arity + 1 {
        return false;
    }

    let mut cur_indices = indices.to_vec();
    let mut cur_values = values.to_vec();

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

        // Sanity: level-0 consistency
        assert!(tree.check_level_consistency(0));

        let root = tree.root();
        let idx = vec![0usize, 3, 7, 11, 54];
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

        // Sanity: consistency across a couple of levels
        assert!(tree.check_level_consistency(0));
        if tree.height() >= 2 {
            assert!(tree.check_level_consistency(1));
        }

        let root = tree.root();
        let idx = vec![0usize, 15, 16, 31, 47, 63];
        let vals: Vec<F> = idx.iter().map(|&i| leaves[i]).collect();
        let proof = tree.open_many(&idx);

        // DS-aware verification with explicit params and tree_label.
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
        // t = 17 (arity = 16), alpha=5, RF=8, RP=64
        let params = poseidon_params_for_width(17);

        // Fixed children and DS fields
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
        assert_eq!(digest1, digest2, "same inputs and DS should be stable");

        // Change DS: level
        let ds_level = DsLabel {
            level: level + 1,
            ..ds
        };
        let d_level = hash_with_ds_dynamic(&ds_level.to_fields(), &children, &params);
        assert_ne!(digest1, d_level, "changing level must change digest");

        // Change DS: position
        let ds_pos = DsLabel {
            position: position + 1,
            ..ds
        };
        let d_pos = hash_with_ds_dynamic(&ds_pos.to_fields(), &children, &params);
        assert_ne!(digest1, d_pos, "changing position must change digest");

        // Change DS: tree_label
        let ds_tree = DsLabel {
            tree_label: tree_label + 1,
            ..ds
        };
        let d_tree = hash_with_ds_dynamic(&ds_tree.to_fields(), &children, &params);
        assert_ne!(digest1, d_tree, "changing tree_label must change digest");

        // Change DS: arity (conceptual change of domain)
        let ds_arity8 = DsLabel { arity: 8, ..ds };
        let d_arity8 = hash_with_ds_dynamic(&ds_arity8.to_fields(), &children, &params);
        assert_ne!(digest1, d_arity8, "changing arity in DS must change digest");

        // Fewer-than-rate children
        let fewer_children: Vec<F> = (0..5).map(|i| F::from(i as u64 + 1)).collect();
        let digest_few_1 = hash_with_ds_dynamic(&ds.to_fields(), &fewer_children, &params);
        let digest_few_2 = hash_with_ds_dynamic(&ds.to_fields(), &fewer_children, &params);
        assert_eq!(digest_few_1, digest_few_2, "stability for partial block");

        // Appending an explicit zero element changes the digest (different message)
        let mut with_extra_zero = fewer_children.clone();
        with_extra_zero.push(F::zero());
        let digest_with_extra = hash_with_ds_dynamic(&ds.to_fields(), &with_extra_zero, &params);
        assert_ne!(
            digest_few_1, digest_with_extra,
            "appending an extra zero element changes the digest (distinct message)"
        );
    }

    #[test]
    fn test_poseidon_params_roundtrip_t9() {
        // t = 9 (arity = 8), alpha=5, RF=8, RP=60
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

        // Change each DS field separately
        let d_level = hash_with_ds_dynamic(
            &DsLabel {
                level: level + 1,
                ..ds
            }
            .to_fields(),
            &children,
            &params,
        );
        assert_ne!(digest1, d_level);

        let d_pos = hash_with_ds_dynamic(
            &DsLabel {
                position: position + 1,
                ..ds
            }
            .to_fields(),
            &children,
            &params,
        );
        assert_ne!(digest1, d_pos);

        let d_tree = hash_with_ds_dynamic(
            &DsLabel {
                tree_label: tree_label + 1,
                ..ds
            }
            .to_fields(),
            &children,
            &params,
        );
        assert_ne!(digest1, d_tree);

        let d_arity16 = hash_with_ds_dynamic(
            &DsLabel { arity: 16, ..ds }.to_fields(),
            &children,
            &params,
        );
        assert_ne!(digest1, d_arity16);

        // Partial-rate case
        let fewer_children: Vec<F> = (0..3).map(|i| F::from(i as u64 + 21)).collect();
        let digest_few = hash_with_ds_dynamic(&ds.to_fields(), &fewer_children, &params);
        let mut with_extra_zero = fewer_children.clone();
        with_extra_zero.push(F::zero());
        let digest_extra = hash_with_ds_dynamic(&ds.to_fields(), &with_extra_zero, &params);
        assert_ne!(digest_few, digest_extra);
    }

    #[test]
    fn merkle_ds_hygiene_negatives_arity16() {
        // Build a small DS-aware arity-16 tree and test DS sensitivity.
        let leaves: Vec<F> = (1..=32).map(|x| F::from(x as u64)).collect();
        let cfg = MerkleChannelCfg::new(16).with_tree_label(1234);
        let tree = MerkleTree::new(leaves.clone(), cfg.clone());

        // Check level-0 consistency: parents of leaves computed via DS-aware hash
        assert!(tree.check_level_consistency(0));

        // Recompute a specific parent at level 0 manually and test DS changes
        let arity = cfg.arity;
        let level0 = 0u32;
        let parent_idx = 1usize; // second parent at level 0
        let base = parent_idx * arity;
        let end = core::cmp::min(base + arity, tree.levels[0].len());
        let children: Vec<F> = tree.levels[0][base..end].iter().map(|w| w.0).collect();

        let ds = DsLabel {
            arity,
            level: level0,
            position: parent_idx as u64,
            tree_label: cfg.tree_label,
        };
        let parent_digest = hash_with_ds_dynamic(&ds.to_fields(), &children, &cfg.params);
        assert_eq!(
            parent_digest, tree.levels[1][parent_idx].0,
            "baseline parent digest matches"
        );

        // Negative: change level
        let d2 = hash_with_ds_dynamic(
            &DsLabel {
                level: level0 + 1,
                ..ds
            }
            .to_fields(),
            &children,
            &cfg.params,
        );
        assert_ne!(parent_digest, d2, "changing level must change digest");

        // Negative: change position
        let d3 = hash_with_ds_dynamic(
            &DsLabel {
                position: (parent_idx as u64) + 1,
                ..ds
            }
            .to_fields(),
            &children,
            &cfg.params,
        );
        assert_ne!(parent_digest, d3, "changing position must change digest");

        // Negative: change tree_label
        let d4 = hash_with_ds_dynamic(
            &DsLabel {
                tree_label: cfg.tree_label + 1,
                ..ds
            }
            .to_fields(),
            &children,
            &cfg.params,
        );
        assert_ne!(parent_digest, d4, "changing tree_label must change digest");

        // Negative: shuffle children order
        let mut shuffled = children.clone();
        if shuffled.len() >= 2 {
            shuffled.swap(0, 1);
        }
        let d5 = hash_with_ds_dynamic(&ds.to_fields(), &shuffled, &cfg.params);
        assert_ne!(parent_digest, d5, "shuffling children must change digest");
    }
}