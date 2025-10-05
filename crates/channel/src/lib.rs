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

// -------------------------
// MLE: core and channel glue
// -------------------------

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}
fn log2_pow2(n: usize) -> usize {
    debug_assert!(is_power_of_two(n));
    usize::BITS as usize - 1 - n.leading_zeros() as usize
}

// A simple multilinear extension over F backed by its table of size 2^k.
#[derive(Clone)]
pub struct Mle {
    table: Vec<F>, // values on {0,1}^k in lexicographic order
    k: usize,
}

impl Mle {
    pub fn new(table: Vec<F>) -> Self {
        assert!(is_power_of_two(table.len()), "MLE length must be 2^k");
        let k = log2_pow2(table.len());
        Self { table, k }
    }

    pub fn from_slice(table: &[F]) -> Self {
        Self::new(table.to_vec())
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn num_vars(&self) -> usize {
        self.k
    }

    pub fn table(&self) -> &[F] {
        &self.table
    }

    // Evaluate at r in F^k using iterative butterfly:
    // fold pairs with weights (1 - r_j, r_j).
    pub fn evaluate(&self, r: &[F]) -> F {
        assert_eq!(r.len(), self.k, "dimension mismatch");
        let mut layer = self.table.clone();
        let mut width = layer.len();
        for (j, &rv) in r.iter().enumerate() {
            let one_minus = F::from(1u64) - rv;
            let half = width / 2;
            for i in 0..half {
                let a = layer[2 * i];
                let b = layer[2 * i + 1];
                layer[i] = one_minus * a + rv * b;
            }
            width = half;
            // no need to zero tail; we will only read first `width` elements next round
            debug_assert_eq!(width, 1 << (self.k - (j + 1)));
        }
        layer[0]
    }
}

// MLE + Merkle commitment helpers over the channel.
pub struct MleProver<'a> {
    merkle: MerkleProver<'a>,
    mle: Mle,
}

pub struct MleVerifier<'a> {
    merkle: MerkleVerifier<'a>,
    k: usize,
}

impl<'a> MleProver<'a> {
    pub fn new(merkle: MerkleProver<'a>, mle: Mle) -> Self {
        Self { merkle, mle }
    }

    // Commit to the MLE table via Merkle and bind root into transcript.
    pub fn commit(&mut self) -> F {
        self.merkle.commit_vector(self.mle.table())
    }

    // Draw k challenges r_0..r_{k-1} from transcript using the provided label.
    pub fn draw_point(&mut self, label: &[u8]) -> Vec<F> {
        // derive r_j as challenge(label || j)
        (0..self.mle.num_vars())
            .map(|j| {
                let mut tag = Vec::with_capacity(label.len() + 8);
                tag.extend_from_slice(label);
                tag.extend_from_slice(&(j as u64).to_le_bytes());
                self.merkle.challenge_scalar(&tag)
            })
            .collect()
    }

    // Evaluate at r and absorb the value into the transcript.
    pub fn evaluate_and_bind(&mut self, r: &[F]) -> F {
        let val = self.mle.evaluate(r);
        self.merkle
            .chan
            .transcript_mut()
            .absorb_bytes(b"MLE/EVAL");
        self.merkle.chan.transcript_mut().absorb_field(val);
        val
    }

    // Optionally, open a batch of indices of the table (e.g., to support later checks).
    pub fn open_indices(&mut self, indices: &[usize]) -> (Vec<F>, MerkleProof) {
        self.merkle.open_indices(indices, self.mle.table())
    }

    pub fn inner_mut(&mut self) -> &mut MerkleProver<'a> {
        &mut self.merkle
    }
}

impl<'a> MleVerifier<'a> {
    pub fn new(merkle: MerkleVerifier<'a>, k: usize) -> Self {
        Self { merkle, k }
    }

    pub fn receive_root(&mut self, root: &F) {
        self.merkle.receive_root(root);
    }

    pub fn draw_point(&mut self, label: &[u8]) -> Vec<F> {
        (0..self.k)
            .map(|j| {
                let mut tag = Vec::with_capacity(label.len() + 8);
                tag.extend_from_slice(label);
                tag.extend_from_slice(&(j as u64).to_le_bytes());
                self.merkle.challenge_scalar(&tag)
            })
            .collect()
    }

    // Bind the claimed evaluation into transcript to mirror prover binding.
    pub fn bind_claimed_eval(&mut self, value: &F) {
        self.merkle
            .chan
            .transcript_mut()
            .absorb_bytes(b"MLE/EVAL");
        self.merkle.chan.transcript_mut().absorb_field(*value);
    }

    // Verify Merkle openings for selected indices of the MLE table.
    pub fn verify_openings(
        &mut self,
        indices: &[usize],
        values: &[F],
        proof: &MerkleProof,
    ) -> bool {
        self.merkle.verify_openings(indices, values, proof)
    }

    pub fn inner_mut(&mut self) -> &mut MerkleVerifier<'a> {
        &mut self.merkle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn e2e_mle_commit_eval_roundtrip() {
        // transcripts
        let params = transcript::default_params();
        let p_tr = Transcript::new(b"MLE-CHAN-E2E", params.clone());
        let v_tr = Transcript::new(b"MLE-CHAN-E2E", params.clone());
        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        // merkle cfg
        let ds_tag = F::from(3030u64);
        let cfg = MerkleChannelCfg::with_default_params(ds_tag);

        // build an MLE with k = 5 (32 entries)
        let mut rng = StdRng::seed_from_u64(999);
        let k = 5usize;
        let n = 1usize << k;
        let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let mlep = Mle::new(table.clone());

        // Prover side: commit
        let mut mp = MerkleProver::new(&mut pchan, cfg.clone());
        let root = mp.commit_vector(&table);

        // Verifier side: receive root
        let mut mv = MerkleVerifier::new(&mut vchan, cfg.clone());
        mv.receive_root(&root);

        // Wrap in MLE helpers
        let mut mle_prover = MleProver::new(mp, mlep.clone());
        let mut mle_verifier = MleVerifier::new(mv, k);

        // Draw the same random point via transcript
        let r_p = mle_prover.draw_point(b"r");
        let r_v = mle_verifier.draw_point(b"r");
        assert_eq!(r_p, r_v);

        // Prover evaluates and binds; verifier binds the claimed value and checks consistency
        let val = mle_prover.evaluate_and_bind(&r_p);
        mle_verifier.bind_claimed_eval(&val);

        // Optional: open a few indices and verify
        let indices = vec![0usize, 1, 2, n - 1];
        let (values, proof) = mle_prover.open_indices(&indices);
        assert!(mle_verifier.verify_openings(&indices, &values, &proof));

        // Sanity: local evaluation matches direct evaluation
        assert_eq!(val, mlep.evaluate(&r_v));
    }
}
