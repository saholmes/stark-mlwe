use field::F;

use commitment::{CommitmentScheme, MerkleCommitment, MerkleConfig, MerkleProof, MerkleRoot};
use transcript::Transcript;

pub struct ProverChannel {
    tr: Transcript,
}

pub struct VerifierChannel {
    tr: Transcript,
}

impl ProverChannel {
    pub fn new(transcript: Transcript) -> Self {
        Self { tr: transcript }
    }

    pub fn transcript_mut(&mut self) -> &mut Transcript {
        &mut self.tr
    }

    pub fn send_digest(&mut self, label: &[u8], digest: &F) {
        self.tr.absorb_bytes(b"CHAN/SEND/DIGEST");
        self.tr.absorb_bytes(label);
        self.tr.absorb_field(*digest);
    }

    pub fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.tr.challenge(label)
    }

    pub fn send_opening(&mut self, indices: &[usize], values: &[F], proof: &MerkleProof) {
        self.tr.absorb_bytes(b"CHAN/SEND/OPEN");
        for &i in indices {
            self.tr
                .absorb_bytes(&u64::try_from(i).expect("index fits u64").to_le_bytes());
        }
        for v in values {
            self.tr.absorb_field(*v);
        }
        self.tr.absorb_bytes(b"PROOF/ARITY");
        self.tr
            .absorb_bytes(&(proof.arity as u64).to_le_bytes());

        self.tr.absorb_bytes(b"PROOF/GROUP_SIZES");
        for lvl in &proof.group_sizes {
            self.tr
                .absorb_bytes(&(lvl.len() as u64).to_le_bytes());
            for sz in lvl {
                self.tr.absorb_bytes(&[*sz]);
            }
        }

        self.tr.absorb_bytes(b"PROOF/SIBLINGS");
        for lvl in &proof.siblings {
            self.tr
                .absorb_bytes(&(lvl.len() as u64).to_le_bytes());
            for s in lvl {
                self.tr.absorb_field(s.0);
            }
        }
    }
}

impl VerifierChannel {
    pub fn new(transcript: Transcript) -> Self {
        Self { tr: transcript }
    }

    pub fn transcript_mut(&mut self) -> &mut Transcript {
        &mut self.tr
    }

    pub fn recv_digest(&mut self, label: &[u8], digest: &F) {
        self.tr.absorb_bytes(b"CHAN/SEND/DIGEST");
        self.tr.absorb_bytes(label);
        self.tr.absorb_field(*digest);
    }

    pub fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.tr.challenge(label)
    }

    pub fn recv_opening(&mut self, indices: &[usize], values: &[F], proof: &MerkleProof) {
        self.tr.absorb_bytes(b"CHAN/SEND/OPEN");
        for &i in indices {
            self.tr
                .absorb_bytes(&u64::try_from(i).expect("index fits u64").to_le_bytes());
        }
        for v in values {
            self.tr.absorb_field(*v);
        }
        self.tr.absorb_bytes(b"PROOF/ARITY");
        self.tr
            .absorb_bytes(&(proof.arity as u64).to_le_bytes());

        self.tr.absorb_bytes(b"PROOF/GROUP_SIZES");
        for lvl in &proof.group_sizes {
            self.tr
                .absorb_bytes(&(lvl.len() as u64).to_le_bytes());
            for sz in lvl {
                self.tr.absorb_bytes(&[*sz]);
            }
        }

        self.tr.absorb_bytes(b"PROOF/SIBLINGS");
        for lvl in &proof.siblings {
            self.tr
                .absorb_bytes(&(lvl.len() as u64).to_le_bytes());
            for s in lvl {
                self.tr.absorb_field(s.0);
            }
        }
    }
}

#[derive(Clone)]
pub struct MerkleChannelCfg {
    pub cfg: MerkleConfig,
}
impl MerkleChannelCfg {
    pub fn new(ds_tag: F, params: poseidon::PoseidonParams) -> Self {
        Self {
            cfg: MerkleConfig::new(ds_tag, params),
        }
    }
    pub fn with_default_params(ds_tag: F) -> Self {
        Self {
            cfg: MerkleConfig::with_default_params(ds_tag),
        }
    }
    fn scheme(&self) -> MerkleCommitment {
        MerkleCommitment::new(self.cfg.clone())
    }
}

pub struct MerkleProver<'a> {
    chan: &'a mut ProverChannel,
    cfg: MerkleChannelCfg,
    root: Option<MerkleRoot>,
    aux: Option<commitment::MerkleAux>,
}

pub struct MerkleVerifier<'a> {
    chan: &'a mut VerifierChannel,
    cfg: MerkleChannelCfg,
    root: Option<MerkleRoot>,
}

impl<'a> MerkleProver<'a> {
    pub fn new(chan: &'a mut ProverChannel, cfg: MerkleChannelCfg) -> Self {
        Self {
            chan,
            cfg,
            root: None,
            aux: None,
        }
    }

    pub fn commit_vector(&mut self, leaves: &[F]) -> F {
        let scheme = self.cfg.scheme();
        let (root, aux) = scheme.commit(leaves);
        self.chan.send_digest(b"commit/root", &root);
        self.root = Some(root);
        self.aux = Some(aux);
        root
    }

    pub fn open_indices(&mut self, indices: &[usize], table: &[F]) -> (Vec<F>, MerkleProof) {
        let values: Vec<F> = indices.iter().map(|&i| table[i]).collect();
        let proof = self
            .cfg
            .scheme()
            .open(indices, self.aux.as_ref().expect("commit first"));
        self.chan.send_opening(indices, &values, &proof);
        (values, proof)
    }

    pub fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.chan.challenge_scalar(label)
    }
}

impl<'a> MerkleVerifier<'a> {
    pub fn new(chan: &'a mut VerifierChannel, cfg: MerkleChannelCfg) -> Self {
        Self {
            chan,
            cfg,
            root: None,
        }
    }

    pub fn receive_root(&mut self, root: &F) {
        self.chan.recv_digest(b"commit/root", root);
        self.root = Some(*root);
    }

    pub fn verify_openings(
        &mut self,
        indices: &[usize],
        values: &[F],
        proof: &MerkleProof,
    ) -> bool {
        self.chan.recv_opening(indices, values, proof);
        let root = self.root.expect("root not set; call receive_root");
        self.cfg.scheme().verify(&root, indices, values, proof)
    }

    pub fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.chan.challenge_scalar(label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn e2e_merkle_channel_roundtrip() {
        // Build transcripts with the same context label and params to match FS
        let params = transcript::default_params();
        let p_tr = Transcript::new(b"MERKLE-CHAN-E2E", params.clone());
        let v_tr = Transcript::new(b"MERKLE-CHAN-E2E", params.clone());

        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        let ds_tag = F::from(2025u64);
        let cfg = MerkleChannelCfg::with_default_params(ds_tag);

        // Prover commits to a random table
        let mut rng = StdRng::seed_from_u64(7);
        let n = 55usize;
        let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let mut prover = MerkleProver::new(&mut pchan, cfg.clone());
        let root = prover.commit_vector(&table);

        let mut verifier = MerkleVerifier::new(&mut vchan, cfg.clone());
        verifier.receive_root(&root);

        // Both sides derive the same challenge via FS
        let alpha_p = prover.challenge_scalar(b"alpha");
        let alpha_v = verifier.challenge_scalar(b"alpha");
        assert_eq!(alpha_p, alpha_v);

        // Prover opens some indices; verifier checks
        let indices = vec![0usize, 3, 7, 11, 54];
        let (values, proof) = prover.open_indices(&indices, &table);
        assert!(verifier.verify_openings(&indices, &values, &proof));
    }
}
