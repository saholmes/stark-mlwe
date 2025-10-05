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

    pub fn mle(&self) -> &Mle {
        &self.mle
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

// -------------------------
// Sum-check over MLE
// -------------------------

// Compute (c0, c1) so that g(t) = c0 + c1 * t, where
// g(t) = sum over remaining variables of f with current fixed prefix and x_i = t.
// When working over the full table layer, this is simply:
// c0 = sum of pairs' left entries, c1 = sum of pairs' (right - left) entries.
fn sumcheck_round_coeffs(layer: &[F]) -> (F, F) {
    // layer length is power-of-two; represents current partial folding domain.
    let mut c0 = F::from(0u64);
    let mut c1 = F::from(0u64);
    for i in (0..layer.len()).step_by(2) {
        let a = layer[i];
        let b = layer[i + 1];
        c0 += a;
        c1 += b - a;
    }
    (c0, c1)
}

pub struct SumCheckProver<'a> {
    mle: MleProver<'a>,
    // Working buffer used to fold the table per round
    layer: Vec<F>,
}

pub struct SumCheckVerifier<'a> {
    mle: MleVerifier<'a>,
    rounds: usize,
}

impl<'a> SumCheckProver<'a> {
    pub fn new(mut mle: MleProver<'a>) -> Self {
        let layer = mle.mle().table().to_vec();
        // Bind claimed sum S into transcript when provided later.
        Self { mle, layer }
    }

    // Start the protocol by committing (already done outside) and sending the claim S.
    // We bind S into the transcript as "SUMCHECK/CLAIM".
    pub fn send_claim(&mut self) -> F {
        let mut s = F::from(0u64);
        for v in &self.layer {
            s += *v;
        }
        self.mle
            .inner_mut()
            .chan
            .transcript_mut()
            .absorb_bytes(b"SUMCHECK/CLAIM");
        self.mle.inner_mut().chan.transcript_mut().absorb_field(s);
        s
    }

    // Run one round: compute g_i(t) = c0 + c1 t, send coefficients,
    // derive r_i from transcript label, and fold layer by r_i.
    pub fn round(&mut self, round_idx: usize, chal_label: &[u8]) -> (F, F, F) {
        debug_assert!(self.layer.len() >= 2);
        let (c0, c1) = sumcheck_round_coeffs(&self.layer);

        // Bind coefficients into transcript in a labeled way
        let t = self.mle.inner_mut().chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/ROUND");
        t.absorb_bytes(&round_idx.to_le_bytes());
        t.absorb_bytes(b"COEFF/c0");
        t.absorb_field(c0);
        t.absorb_bytes(b"COEFF/c1");
        t.absorb_field(c1);

        // Fiat–Shamir challenge r_i
        let mut label = Vec::with_capacity(chal_label.len() + 8);
        label.extend_from_slice(chal_label);
        label.extend_from_slice(&(round_idx as u64).to_le_bytes());
        let r_i = self.mle.inner_mut().chan.challenge_scalar(&label);

        // Fold layer by r_i: layer'[j] = (1-r_i) * layer[2j] + r_i * layer[2j+1]
        let one_minus = F::from(1u64) - r_i;
        let mut j = 0usize;
        for i in 0..(self.layer.len() / 2) {
            let a = self.layer[j];
            let b = self.layer[j + 1];
            self.layer[i] = one_minus * a + r_i * b;
            j += 2;
        }
        self.layer.truncate(self.layer.len() / 2);

        (c0, c1, r_i)
    }

    // Finalize by sending the evaluation at r (already implied by the folded layer):
    // After k rounds, layer has length 1 and equals f(r).
    pub fn finalize_and_bind_eval(&mut self) -> F {
        debug_assert_eq!(self.layer.len(), 1);
        let val = self.layer[0];
        self.mle
            .inner_mut()
            .chan
            .transcript_mut()
            .absorb_bytes(b"SUMCHECK/FINAL/EVAL");
        self.mle.inner_mut().chan.transcript_mut().absorb_field(val);
        val
    }

    pub fn mle_prover_mut(&mut self) -> &mut MleProver<'a> {
        &mut self.mle
    }
}

impl<'a> SumCheckVerifier<'a> {
    pub fn new(mle: MleVerifier<'a>) -> Self {
        let rounds = mle.k;
        Self { mle, rounds }
    }

    // Receive and bind the claimed sum S.
    pub fn recv_claim(&mut self, s: &F) {
        let t = self.mle.inner_mut().chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/CLAIM");
        t.absorb_field(*s);
    }

    // One round verification:
    // - Verify g_i(0) + g_i(1) equals the running sum S_prev.
    // - Derive r_i via FS using the same label scheme.
    // - Update running sum to S := c0 + c1 * r_i.
    pub fn round(
        &mut self,
        round_idx: usize,
        s_prev: F,
        c0: F,
        c1: F,
        chal_label: &[u8],
    ) -> (F, F) {
        // Bind coefficients (mirror prover)
        let t = self.mle.inner_mut().chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/ROUND");
        t.absorb_bytes(&round_idx.to_le_bytes());
        t.absorb_bytes(b"COEFF/c0");
        t.absorb_field(c0);
        t.absorb_bytes(b"COEFF/c1");
        t.absorb_field(c1);

        // Check g(0) + g(1) = (c0) + (c0 + c1) = 2*c0 + c1
        let lhs = F::from(2u64) * c0 + c1;
        assert_eq!(lhs, s_prev, "sum-check round consistency failed");

        // Challenge
        let mut label = Vec::with_capacity(chal_label.len() + 8);
        label.extend_from_slice(chal_label);
        label.extend_from_slice(&(round_idx as u64).to_le_bytes());
        let r_i = self.mle.inner_mut().chan.challenge_scalar(&label);

        // Update claim
        let s_next = c0 + c1 * r_i;
        (r_i, s_next)
    }

    // Final check: bind the received final evaluation and assert it equals f(r) claim S_k.
    pub fn finalize_and_check(&mut self, eval_at_r: F, s_k: F) {
        let t = self.mle.inner_mut().chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/FINAL/EVAL");
        t.absorb_field(eval_at_r);
        assert_eq!(eval_at_r, s_k, "final sum-check evaluation mismatch");
    }

    pub fn mle_verifier_mut(&mut self) -> &mut MleVerifier<'a> {
        &mut self.mle
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

        // Prover opens a set of indices; verifier checks the opening
        let indices = vec![0usize, 3, 7, 11, 54];
        let (values, proof) = prover.open_indices(&indices, &table);
        assert!(verifier.verify_openings(&indices, &values, &proof));
    }

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
        let mut mle_prover = super::MleProver::new(mp, mlep.clone());
        let mut mle_verifier = super::MleVerifier::new(mv, k);

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

    #[test]
    fn e2e_sumcheck_roundtrip() {
        // transcripts
        let params = transcript::default_params();
        let p_tr = Transcript::new(b"SUMCHECK-E2E", params.clone());
        let v_tr = Transcript::new(b"SUMCHECK-E2E", params.clone());
        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        // merkle cfg
        let ds_tag = F::from(5050u64);
        let cfg = MerkleChannelCfg::with_default_params(ds_tag);

        // build an MLE with k = 6 (64 entries)
        let mut rng = StdRng::seed_from_u64(42);
        let k = 6usize;
        let n = 1usize << k;
        let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let mle = Mle::new(table.clone());

        // Prover commit
        let mut mp = MerkleProver::new(&mut pchan, cfg.clone());
        let root = mp.commit_vector(&table);

        // Verifier receives root
        let mut mv = MerkleVerifier::new(&mut vchan, cfg.clone());
        mv.receive_root(&root);

        // Wrap in MLE helpers
        let mle_p = MleProver::new(mp, mle.clone());
        let mle_v = MleVerifier::new(mv, k);

        let mut sp = SumCheckProver::new(mle_p);
        let mut sv = SumCheckVerifier::new(mle_v);

        // Prover sends claim S
        let s = sp.send_claim();
        sv.recv_claim(&s);

        // Run k rounds
        let mut running = s;
        for i in 0..k {
            let (c0, c1, r_i) = sp.round(i, b"sumcheck/r");
            let (r_i_v, s_next) = sv.round(i, running, c0, c1, b"sumcheck/r");
            assert_eq!(r_i, r_i_v, "challenge mismatch at round {}", i);
            running = s_next;
        }

        // Final evaluation
        let eval = sp.finalize_and_bind_eval();
        sv.finalize_and_check(eval, running);
    }
}
