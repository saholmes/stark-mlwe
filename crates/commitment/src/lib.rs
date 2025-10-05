//! Commitment abstraction with a Merkle implementation matched to your merkle crate.

use ark_pallas::Fr as F;
use poseidon::{params::generate_params_t17_x5, PoseidonParams};

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
    pub ds_tag: F,
    pub params: PoseidonParams,
}
impl MerkleConfig {
    pub fn new(ds_tag: F, params: PoseidonParams) -> Self {
        Self { ds_tag, params }
    }
    pub fn with_default_params(ds_tag: F) -> Self {
        Self {
            ds_tag,
            params: default_params(),
        }
    }
}

/// Deterministic default Poseidon params consistent with your merkle crate.
pub fn default_params() -> PoseidonParams {
    let seed = b"POSEIDON-T17-X5-SEED";
    generate_params_t17_x5(seed)
}

// Import merkle types.
pub use merkle::{default_params as merkle_default_params, MerkleProof, MerkleTree};

/// The commitment digest type is a field element F (root()).
pub type MerkleRoot = F;

/// Prover-side auxiliary data: we keep the whole MerkleTree instance.
#[derive(Clone)]
pub struct MerkleAux {
    pub tree: MerkleTree,
}

/// Merkle commitment scheme adapter.
pub struct MerkleCommitment {
    cfg: MerkleConfig,
}

impl MerkleCommitment {
    pub fn new(cfg: MerkleConfig) -> Self {
        Self { cfg }
    }
}

impl CommitmentScheme for MerkleCommitment {
    type Digest = MerkleRoot;
    type Proof = MerkleProof;
    type Aux = MerkleAux;

    fn commit(&self, leaves: &[F]) -> (Self::Digest, Self::Aux) {
        let tree = MerkleTree::new(leaves.to_vec(), self.cfg.ds_tag, self.cfg.params.clone());
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
        merkle::verify_many(
            root,
            indices,
            values,
            proof,
            self.cfg.ds_tag,
            self.cfg.params.clone(),
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
        // Choose a non-power-of-arity size to exercise partial groups
        let n = 37usize;
        let leaves: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let cfg = MerkleConfig::with_default_params(F::from(123u64));
        let scheme = MerkleCommitment::new(cfg.clone());
        let (root, aux) = scheme.commit(&leaves);

        let query_indices = vec![0usize, 5, 7, 16, 36];
        let proof = scheme.open(&query_indices, &aux);

        // Select the corresponding values in the same order as query_indices
        let query_values: Vec<F> = query_indices.iter().map(|&i| leaves[i]).collect();

        assert!(scheme.verify(&root, &query_indices, &query_values, &proof));
    }
}
