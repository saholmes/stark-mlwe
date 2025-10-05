//! Commitment abstraction with a Merkle implementation matched to your merkle crate.

use field::F;
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
    // Matches merkle::default_params() seed
    let seed = b"POSEIDON-T17-X5-SEED";
    generate_params_t17_x5(seed)
}

// Import merkle types exactly as provided.
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

    fn open(&self, _indices: &[usize], _aux: &Self::Aux) -> Self::Proof {
        // TODO: Wire to your Merkle proof construction method when available.
        // For example, if you add:
        //   impl MerkleTree { pub fn open_many(&self, indices: &[usize]) -> MerkleProof { ... } }
        // then replace this with: aux.tree.open_many(indices)
        unimplemented!("MerkleTree::open_many is not implemented in the merkle crate yet")
    }

    fn verify(
        &self,
        _root: &Self::Digest,
        _indices: &[usize],
        _values: &[F],
        _proof: &Self::Proof,
    ) -> bool {
        // TODO: Wire to your Merkle verification when available.
        // e.g., merkle::verify_many(root, indices, values, proof, self.cfg.ds_tag, self.cfg.params.clone())
        unimplemented!("merkle::verify_many is not implemented in the merkle crate yet")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;

    #[test]
    fn merkle_commit_builds_root() {
        let mut rng = ark_std::test_rng();
        let n = 16usize;
        let leaves: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let cfg = MerkleConfig::with_default_params(F::from(123u64));
        let scheme = MerkleCommitment::new(cfg);
        let (root, aux) = scheme.commit(&leaves);

        // Root matches the tree’s root.
        assert_eq!(root, aux.tree.root());

        // Opening is not implemented yet; ensure the adapter compiles and basic commit works.
        // Once you add open/verify in the merkle crate, we will enable the full round-trip test.
    }
}
