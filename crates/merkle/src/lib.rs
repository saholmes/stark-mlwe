use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use field::F;
use poseidon::{hash_with_ds, params::generate_params_t17_x5, PoseidonParams};
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

#[derive(Clone, Serialize, Deserialize)]
pub struct MerkleTree {
    pub leaves: Vec<SerFr>,
    pub root: SerFr,
    pub ds_tag: SerFr,
    // level 0 = leaves (as digests), higher levels are parent digests
    pub levels: Vec<Vec<SerFr>>,
    // We skip serializing PoseidonParams; on deserialize, fill it via default_params().
    #[serde(skip, default = "default_params")]
    pub params: PoseidonParams,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MerkleProof {
    pub indices: Vec<usize>,
    pub siblings: Vec<Vec<SerFr>>,
    pub group_sizes: Vec<Vec<u8>>,
    pub arity: usize,
}

impl MerkleTree {
    pub fn new(leaves: Vec<F>, ds_tag: F, params: PoseidonParams) -> Self {
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
        }
    }

    pub fn root(&self) -> F {
        self.root.0
    }

    pub fn arity(&self) -> usize {
        poseidon::RATE
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
}

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

    for (level_siblings, level_group_sizes) in proof.siblings.iter().zip(proof.group_sizes.iter())
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

        // Per-level sibling cursor reset to 0
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

            let parent_digest = hash_with_ds(&children, ds_tag, &params);
            next_indices.push(parent_idx);
            next_values.push(parent_digest);
        }

        // Ensure we consumed exactly this level's siblings
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
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn merkle_proof_roundtrip_arbitrary_size() {
        let mut rng = StdRng::seed_from_u64(123);
        let n = 55usize;
        let leaves: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let params = default_params();
        let ds = F::from(77u64);
        let tree = MerkleTree::new(leaves.clone(), ds, params.clone());
        let root = tree.root();

        let idx = vec![0usize, 3, 7, 11, 54];
        let vals: Vec<F> = idx.iter().map(|&i| leaves[i]).collect();
        let proof = tree.open_many(&idx);
        assert!(verify_many(&root, &idx, &vals, &proof, ds, params));
    }
}
