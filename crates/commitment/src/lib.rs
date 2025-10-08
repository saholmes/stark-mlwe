//! Commitment abstraction with a Merkle implementation matched to your merkle crate.

use ark_pallas::Fr as F;

// Poseidon params
use poseidon::{params::generate_params_t17_x5, PoseidonParams, PoseidonParamsDynamic};
use poseidon::dynamic_from_static_t17; // adapter

// Import merkle types.
pub use merkle::{verify_many_ds, MerkleChannelCfg, MerkleProof, MerkleTree};

/// Trait for vector commitments over field elements.
pub trait CommitmentScheme {
    type Digest: Clone;
    type Proof: Clone;
    type Aux: Clone;

    fn commit(&self, leaves: &[F]) -> (Self::Digest, Self::Aux);
    fn open(&self, indices: &[usize], aux: &Self::Aux) -> Self::Proof;
    fn verify(
        &self,
        root: &Self::Digest,
        indices: &[usize],
        values: &[F],
        proof: &Self::Proof,
    ) -> bool;
}

/// Configuration for Merkle commitments.
#[derive(Clone)]
pub struct MerkleConfig {
    pub ds_tag: u64,           // tree_label for DS-aware hashing
    pub params: PoseidonParams // static t=17 params kept for compatibility if needed
}
impl MerkleConfig {
    pub fn new(ds_tag: u64, params: PoseidonParams) -> Self {
        Self { ds_tag, params }
    }
    pub fn with_default_params(ds_tag: u64) -> Self {
        Self {
            ds_tag,
            params: default_params(),
        }
    }
}

/// Deterministic default Poseidon params (t=17).
pub fn default_params() -> PoseidonParams {
    let seed = b"POSEIDON-T17-X5-SEED";
    generate_params_t17_x5(seed)
}

pub type MerkleRoot = F;

#[derive(Clone)]
pub struct MerkleAux {
    pub tree: MerkleTree,
}

pub struct MerkleCommitment {
    cfg: MerkleConfig,
}

impl MerkleCommitment {
    pub fn new(cfg: MerkleConfig) -> Self {
        Self { cfg }
    }

    fn tree_cfg(&self) -> MerkleChannelCfg {
        // Use arity=16 (t=17) and DS-aware dynamic params.
        let dyn_params: PoseidonParamsDynamic = dynamic_from_static_t17(&self.cfg.params);
        MerkleChannelCfg {
            arity: 16,
            tree_label: self.cfg.ds_tag,
            params: dyn_params,
        }
    }
}

impl CommitmentScheme for MerkleCommitment {
    type Digest = MerkleRoot;
    type Proof = MerkleProof;
    type Aux = MerkleAux;

    fn commit(&self, leaves: &[F]) -> (Self::Digest, Self::Aux) {
        let cfg = self.tree_cfg();
        let tree = MerkleTree::new(leaves.to_vec(), cfg);
        let root = tree.root();
        (root, MerkleAux { tree })
    }

    fn open(&self, indices: &[usize], aux: &Self::Aux) -> Self::Proof {
        aux.tree.open_many(indices)
    }

    fn verify(
        &self,
        root: &Self::Digest,
        indices: &[usize],
        values: &[F],
        proof: &Self::Proof,
    ) -> bool {
        // DS-aware verification to match tree construction.
        let dyn_params: PoseidonParamsDynamic = dynamic_from_static_t17(&self.cfg.params);
        verify_many_ds(
            root,
            indices,
            values,
            proof,
            self.cfg.ds_tag, // tree_label
            dyn_params,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn merkle_commit_open_verify_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 64usize; // multiple of 16 to exercise full groups
        let leaves: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let cfg = MerkleConfig::with_default_params(123u64);
        let scheme = MerkleCommitment::new(cfg.clone());
        let (root, aux) = scheme.commit(&leaves);

        let query_indices = vec![0usize, 15, 16, 31, 47, 63];
        let proof = scheme.open(&query_indices, &aux);
        let query_values: Vec<F> = query_indices.iter().copied().map(|i| leaves[i]).collect();

        assert!(scheme.verify(&root, &query_indices, &query_values, &proof));
    }
}