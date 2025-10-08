use ark_pallas::Fr as F;

use ark_ff::{BigInteger, PrimeField};
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

// Helper: map an Fr tag to a u64 tree_label deterministically (low 64 bits).
fn fr_tag_to_u64(tag: F) -> u64 {
    let bi = tag.into_bigint();
    // ark_ff BigInteger exposes limbs via internal array; low limb is index 0 (little-endian).
    bi.0[0] as u64
}

#[derive(Clone)]
pub struct MerkleChannelCfg {
    pub cfg: MerkleConfig,
}
impl MerkleChannelCfg {
    // Accept Fr for compatibility and convert to u64 tree_label for DS-aware Merkle.
    pub fn new(ds_tag: F, params: poseidon::PoseidonParams) -> Self {
        let tree_label = fr_tag_to_u64(ds_tag);
        Self {
            cfg: MerkleConfig::new(tree_label, params),
        }
    }
    pub fn with_default_params(ds_tag: F) -> Self {
        let tree_label = fr_tag_to_u64(ds_tag);
        Self {
            cfg: MerkleConfig::with_default_params(tree_label),
        }
    }
    fn scheme(&self) -> MerkleCommitment {
        MerkleCommitment::new(self.cfg.clone())
    }
}

pub struct MerkleProver <'a> {
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

    pub fn root(&self) -> Option<F> {
        self.root
    }

    pub fn aux(&self) -> Option<&commitment::MerkleAux> {
        self.aux.as_ref()
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
        self.root
            .map(|root| self.cfg.scheme().verify(&root, indices, values, proof))
            .unwrap_or(false)
    }

    pub fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.chan.challenge_scalar(label)
    }

    pub fn root(&self) -> Option<F> {
        self.root
    }
}

// -------------------------
// MLE core
// -------------------------

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}
fn log2_pow2(n: usize) -> usize {
    debug_assert!(is_power_of_two(n));
    usize::BITS as usize - 1 - n.leading_zeros() as usize
}

#[derive(Clone)]
pub struct Mle {
    table: Vec<F>,
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

// -------------------------
// MLE + Merkle helpers
// -------------------------

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

    pub fn commit(&mut self) -> F {
        self.merkle.commit_vector(self.mle.table())
    }

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

    pub fn evaluate_and_bind(&mut self, r: &[F]) -> F {
        let val = self.mle.evaluate(r);
        self.merkle
            .chan
            .transcript_mut()
            .absorb_bytes(b"MLE/EVAL");
        self.merkle.chan.transcript_mut().absorb_field(val);
        val
    }

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

    pub fn bind_claimed_eval(&mut self, value: &F) {
        self.merkle
            .chan
            .transcript_mut()
            .absorb_bytes(b"MLE/EVAL");
        self.merkle.chan.transcript_mut().absorb_field(*value);
    }

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

    pub fn k(&self) -> usize {
        self.k
    }
}

// -------------------------
// Sum-check (plain)
// -------------------------

fn sumcheck_round_coeffs(layer: &[F]) -> (F, F) {
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
    layer: Vec<F>,
}

pub struct SumCheckVerifier<'a> {
    mle: MleVerifier<'a>,
}

impl<'a> SumCheckProver<'a> {
    pub fn new(mle: MleProver<'a>) -> Self {
        let layer = mle.mle().table().to_vec();
        Self { mle, layer }
    }

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

    pub fn round(&mut self, round_idx: usize, chal_label: &[u8]) -> (F, F, F) {
        debug_assert!(self.layer.len() >= 2);
        let (c0, c1) = sumcheck_round_coeffs(&self.layer);

        let t = self.mle.inner_mut().chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/ROUND");
        t.absorb_bytes(&round_idx.to_le_bytes());
        t.absorb_bytes(b"COEFF/c0");
        t.absorb_field(c0);
        t.absorb_bytes(b"COEFF/c1");
        t.absorb_field(c1);

        let mut label = Vec::with_capacity(chal_label.len() + 8);
        label.extend_from_slice(chal_label);
        label.extend_from_slice(&(round_idx as u64).to_le_bytes());
        let r_i = self.mle.inner_mut().chan.challenge_scalar(&label);

        let one_minus = F::from(1u64) - r_i;
        for j in 0..(self.layer.len() / 2) {
            let a = self.layer[2 * j];
            let b = self.layer[2 * j + 1];
            self.layer[j] = one_minus * a + r_i * b;
        }
        self.layer.truncate(self.layer.len() / 2);

        (c0, c1, r_i)
    }

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
        Self { mle }
    }

    pub fn recv_claim(&mut self, s: &F) {
        let t = self.mle.inner_mut().chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/CLAIM");
        t.absorb_field(*s);
    }

    pub fn round(
        &mut self,
        round_idx: usize,
        s_prev: F,
        c0: F,
        c1: F,
        chal_label: &[u8],
    ) -> (F, F) {
        let t = self.mle.inner_mut().chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/ROUND");
        t.absorb_bytes(&round_idx.to_le_bytes());
        t.absorb_bytes(b"COEFF/c0");
        t.absorb_field(c0);
        t.absorb_bytes(b"COEFF/c1");
        t.absorb_field(c1);

        let lhs = F::from(2u64) * c0 + c1;
        assert_eq!(lhs, s_prev, "sum-check round consistency failed");

        let mut label = Vec::with_capacity(chal_label.len() + 8);
        label.extend_from_slice(chal_label);
        label.extend_from_slice(&(round_idx as u64).to_le_bytes());
        let r_i = self.mle.inner_mut().chan.challenge_scalar(&label);

        let s_next = c0 + c1 * r_i;
        (r_i, s_next)
    }

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

// -------------------------
// Merkle-folded sum-check
// -------------------------

#[derive(Clone, Copy)]
pub struct SumCheckMFConfig {
    pub queries_per_round: usize, // q
}

impl Default for SumCheckMFConfig {
    fn default() -> Self {
        Self { queries_per_round: 2 }
    }
}

struct FoldedLayer {
    values: Vec<F>,
    aux: commitment::MerkleAux,
    root: F,
}

pub struct SumCheckMFProver<'a> {
    cfg: SumCheckMFConfig,
    merkle_cfg: MerkleChannelCfg,
    chan: &'a mut ProverChannel,
    scheme: MerkleCommitment,
    cur: FoldedLayer,
    rounds: usize,
}

pub struct SumCheckMFVerifier<'a> {
    cfg: SumCheckMFConfig,
    merkle_cfg: MerkleChannelCfg,
    chan: &'a mut VerifierChannel,
    scheme: MerkleCommitment,
    cur_root: F,
    rounds: usize,
}

pub struct MFFoldOpenings {
    pub cur_indices: Vec<usize>,
    pub cur_values: Vec<F>,
    pub cur_proof: MerkleProof,
    pub next_indices: Vec<usize>,
    pub next_values: Vec<F>,
    pub next_proof: MerkleProof,
}

// Deterministic r_i from only (round index, prev_root) using a fresh temporary transcript.
fn mf_round_challenge_from_root(round_idx: usize, prev_root: &F, tr_params: &poseidon::PoseidonParams) -> F {
    let mut tmp = Transcript::new(b"SUMCHECK-MF/ROUND-CHAL", tr_params.clone());
    tmp.absorb_bytes(b"SUMCHECK/MF/R");
    tmp.absorb_bytes(&u64::try_from(round_idx).unwrap().to_le_bytes());
    tmp.absorb_field(*prev_root);
    tmp.challenge(b"r_i")
}

impl<'a> SumCheckMFProver<'a> {
    pub fn new(
        cfg: SumCheckMFConfig,
        merkle_cfg: MerkleChannelCfg,
        chan: &'a mut ProverChannel,
        mle: &Mle,
    ) -> Self {
        let scheme = merkle_cfg.scheme();
        let (root, aux) = scheme.commit(mle.table());
        chan.send_digest(b"sumcheck-mf/root/0", &root);
        let cur = FoldedLayer {
            values: mle.table().to_vec(),
            aux,
            root,
        };
        Self {
            cfg,
            merkle_cfg,
            chan,
            scheme,
            cur,
            rounds: mle.num_vars(),
        }
    }

    pub fn send_claim(&mut self) -> F {
        let s = self.cur.values.iter().copied().fold(F::from(0u64), |acc, x| acc + x);
        self.chan.transcript_mut().absorb_bytes(b"SUMCHECK/MF/CLAIM");
        self.chan.transcript_mut().absorb_field(s);
        s
    }

    pub fn round(&mut self, i: usize) -> (F, F, F, F, MFFoldOpenings) {
        let (c0, c1) = sumcheck_round_coeffs(&self.cur.values);

        let t = self.chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/MF/ROUND");
        t.absorb_bytes(&i.to_le_bytes());
        t.absorb_bytes(b"COEFF/c0");
        t.absorb_field(c0);
        t.absorb_bytes(b"COEFF/c1");
        t.absorb_field(c1);

        let r_i = {
            let params = self.chan.transcript_mut().params().clone();
            mf_round_challenge_from_root(i, &self.cur.root, &params)
        };

        let one_minus = F::from(1u64) - r_i;
        let half = self.cur.values.len() / 2;
        let mut next = Vec::with_capacity(half);
        for j in 0..half {
            let a = self.cur.values[2 * j];
            let b = self.cur.values[2 * j + 1];
            next.push(one_minus * a + r_i * b);
        }
        let (next_root, next_aux) = self.scheme.commit(&next);
        self.chan.send_digest(b"sumcheck-mf/root/next", &next_root);

        // sample unique, sorted queries Q_i
        use std::collections::BTreeSet;
        let q_target = core::cmp::min(self.cfg.queries_per_round.max(1), half);
        let mut set = BTreeSet::new();
        let mut attempt = 0usize;
        let max_attempts = q_target.saturating_mul(16).max(16);
        let mut j = 0usize;
        while set.len() < q_target && attempt < max_attempts {
            let mut qlabel = Vec::new();
            qlabel.extend_from_slice(b"sumcheck-mf/q");
            qlabel.extend_from_slice(&i.to_le_bytes());
            qlabel.extend_from_slice(&(j as u64).to_le_bytes());
            use ark_ff::PrimeField;
            let r = self.chan.challenge_scalar(&qlabel);
            let bytes = r.into_bigint().to_bytes_le();
            let mut acc = 0u64;
            for chunk in bytes.chunks(8) {
                let mut le = [0u8; 8];
                le[..chunk.len()].copy_from_slice(chunk);
                acc ^= u64::from_le_bytes(le);
            }
            if half > 0 {
                let idx = (acc as usize) % half;
                set.insert(idx);
            }
            j += 1;
            attempt += 1;
        }
        // Fallback: deterministically fill remaining indices by scanning
        if set.len() < q_target {
            for idx in 0..half {
                set.insert(idx);
                if set.len() == q_target {
                    break;
                }
            }
        }
        let queries: Vec<usize> = set.into_iter().collect();

        // openings
        let mut cur_indices: Vec<usize> = Vec::with_capacity(2 * queries.len());
        for &jj in &queries {
            cur_indices.push(2 * jj);
            cur_indices.push(2 * jj + 1);
        }
        let cur_values: Vec<F> = cur_indices.iter().map(|&ix| self.cur.values[ix]).collect();
        let cur_proof = self.scheme.open(&cur_indices, &self.cur.aux);

        let next_indices = queries.clone();
        let next_values: Vec<F> = next_indices.iter().map(|&ix| next[ix]).collect();
        let next_proof = self.scheme.open(&next_indices, &next_aux);

        self.chan.send_opening(&cur_indices, &cur_values, &cur_proof);
        self.chan
            .send_opening(&next_indices, &next_values, &next_proof);

        self.cur = FoldedLayer {
            values: next,
            aux: next_aux,
            root: next_root,
        };

        let openings = MFFoldOpenings {
            cur_indices,
            cur_values,
            cur_proof,
            next_indices,
            next_values,
            next_proof,
        };

        (c0, c1, r_i, self.cur.root, openings)
    }

    pub fn finalize_eval(&mut self) -> F {
        debug_assert_eq!(self.cur.values.len(), 1);
        let val = self.cur.values[0];
        self.chan.transcript_mut().absorb_bytes(b"SUMCHECK/MF/FINAL/EVAL");
        self.chan.transcript_mut().absorb_field(val);
        val
    }
}

impl<'a> SumCheckMFVerifier<'a> {
    pub fn new(
        cfg: SumCheckMFConfig,
        merkle_cfg: MerkleChannelCfg,
        chan: &'a mut VerifierChannel,
        initial_root: F,
        rounds: usize,
    ) -> Self {
        let scheme = merkle_cfg.scheme();
        Self {
            cfg,
            merkle_cfg,
            chan,
            scheme,
            cur_root: initial_root,
            rounds,
        }
    }

    pub fn receive_initial_root(&mut self, root: &F) {
        self.chan.recv_digest(b"sumcheck-mf/root/0", root);
        self.cur_root = *root;
    }

    pub fn recv_claim(&mut self, s: &F) {
        let t = self.chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/MF/CLAIM");
        t.absorb_field(*s);
    }

    pub fn start_round(
        &mut self,
        i: usize,
        s_prev: F,
        c0: F,
        c1: F,
    ) {
        let t = self.chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/MF/ROUND");
        t.absorb_bytes(&i.to_le_bytes());
        t.absorb_bytes(b"COEFF/c0");
        t.absorb_field(c0);
        t.absorb_bytes(b"COEFF/c1");
        t.absorb_field(c1);

        let lhs = F::from(2u64) * c0 + c1;
        assert_eq!(lhs, s_prev, "sum-check MF round consistency failed");
    }

    pub fn derive_round_challenge(&mut self, i: usize) -> F {
        let params = self.chan.transcript_mut().params().clone();
        mf_round_challenge_from_root(i, &self.cur_root, &params)
    }

    pub fn recv_next_root(&mut self, next_root: F) {
        self.chan.recv_digest(b"sumcheck-mf/root/next", &next_root);
        self.cur_root = next_root;
    }

    pub fn compute_s_next(&self, c0: F, c1: F, r_i: F) -> F {
        c0 + c1 * r_i
    }

    pub fn verify_fold_openings(
        &mut self,
        cur_indices: &[usize],
        cur_values: &[F],
        cur_proof: &MerkleProof,
        next_indices: &[usize],
        next_values: &[F],
        next_proof: &MerkleProof,
        r_i: F,
        prev_root: F,
        next_root: F,
    ) -> bool {
        let ok_cur = self
            .scheme
            .verify(&prev_root, cur_indices, cur_values, cur_proof);
        if !ok_cur {
            return false;
        }
        let ok_next = self
            .scheme
            .verify(&next_root, next_indices, next_values, next_proof);
        if !ok_next {
            return false;
        }
        use std::collections::BTreeMap;
        if cur_indices.len() != cur_values.len() || next_indices.len() != next_values.len() {
            return false;
        }
        let mut pairs: BTreeMap<usize, (Option<F>, Option<F>)> = BTreeMap::new();
        for (&ix, &val) in cur_indices.iter().zip(cur_values.iter()) {
            let j = ix / 2;
            if ix % 2 == 0 {
                pairs.entry(j).or_default().0 = Some(val);
            } else {
                pairs.entry(j).or_default().1 = Some(val);
            }
        }
        let one_minus = F::from(1u64) - r_i;
        for (&j, &vj) in next_indices.iter().zip(next_values.iter()) {
            let (a_opt, b_opt) = pairs.get(&j).copied().unwrap_or((None, None));
            let (a, b) = match (a_opt, b_opt) {
                (Some(a), Some(b)) => (a, b),
                _ => return false,
            };
            let folded = one_minus * a + r_i * b;
            if folded != vj {
                return false;
            }
        }
        true
    }

    pub fn finalize_and_check(&mut self, final_eval: F, s_k: F) {
        let t = self.chan.transcript_mut();
        t.absorb_bytes(b"SUMCHECK/MF/FINAL/EVAL");
        t.absorb_field(final_eval);
        assert_eq!(final_eval, s_k, "final MF sum-check evaluation mismatch");
    }

    pub fn current_root(&self) -> F {
        self.cur_root
    }
}

// -------------------------
// Tests
// -------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn e2e_merkle_channel_roundtrip() {
        let params = transcript::default_params();
        let p_tr = Transcript::new(b"MERKLE-CHAN-E2E", params.clone());
        let v_tr = Transcript::new(b"MERKLE-CHAN-E2E", params.clone());

        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        let ds_tag = F::from(2025u64);
        let cfg = MerkleChannelCfg::with_default_params(ds_tag);

        let mut rng = StdRng::seed_from_u64(7);
        let n = 55usize;
        let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let mut prover = MerkleProver::new(&mut pchan, cfg.clone());
        let root = prover.commit_vector(&table);

        let mut verifier = MerkleVerifier::new(&mut vchan, cfg.clone());
        verifier.receive_root(&root);

        let alpha_p = prover.challenge_scalar(b"alpha");
        let alpha_v = verifier.challenge_scalar(b"alpha");
        assert_eq!(alpha_p, alpha_v);

        let indices = vec![0usize, 3, 7, 11, 54];
        let (values, proof) = prover.open_indices(&indices, &table);
        assert!(verifier.verify_openings(&indices, &values, &proof));
    }

    #[test]
    fn e2e_mle_commit_eval_roundtrip() {
        let params = transcript::default_params();
        let p_tr = Transcript::new(b"MLE-CHAN-E2E", params.clone());
        let v_tr = Transcript::new(b"MLE-CHAN-E2E", params.clone());
        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        let ds_tag = F::from(3030u64);
        let cfg = MerkleChannelCfg::with_default_params(ds_tag);

        let mut rng = StdRng::seed_from_u64(999);
        let k = 5usize;
        let n = 1usize << k;
        let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let mlep = Mle::new(table.clone());

        let mut mp = MerkleProver::new(&mut pchan, cfg.clone());
        let root = mp.commit_vector(&table);

        let mut mv = MerkleVerifier::new(&mut vchan, cfg.clone());
        mv.receive_root(&root);

        let mut mle_prover = super::MleProver::new(mp, mlep.clone());
        let mut mle_verifier = super::MleVerifier::new(mv, k);

        let r_p = mle_prover.draw_point(b"r");
        let r_v = mle_verifier.draw_point(b"r");
        assert_eq!(r_p, r_v);

        let val = mle_prover.evaluate_and_bind(&r_p);
        mle_verifier.bind_claimed_eval(&val);

        let indices = vec![0usize, 1, 2, n - 1];
        let (values, proof) = mle_prover.open_indices(&indices);
        assert!(mle_verifier.verify_openings(&indices, &values, &proof));

        assert_eq!(val, mlep.evaluate(&r_v));
    }

    #[test]
    fn e2e_sumcheck_roundtrip() {
        let params = transcript::default_params();
        let p_tr = Transcript::new(b"SUMCHECK-E2E", params.clone());
        let v_tr = Transcript::new(b"SUMCHECK-E2E", params.clone());
        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        let ds_tag = F::from(5050u64);
        let cfg = MerkleChannelCfg::with_default_params(ds_tag);

        let mut rng = StdRng::seed_from_u64(42);
        let k = 6usize;
        let n = 1usize << k;
        let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let mle = Mle::new(table.clone());

        let mut mp = MerkleProver::new(&mut pchan, cfg.clone());
        let root = mp.commit_vector(&table);

        let mut mv = MerkleVerifier::new(&mut vchan, cfg.clone());
        mv.receive_root(&root);

        let mle_p = MleProver::new(mp, mle.clone());
        let mle_v = MleVerifier::new(mv, k);

        let mut sp = SumCheckProver::new(mle_p);
        let mut sv = SumCheckVerifier::new(mle_v);

        let s = sp.send_claim();
        sv.recv_claim(&s);

        let mut running = s;
        for i in 0..k {
            let (c0, c1, r_i) = sp.round(i, b"sumcheck/r");
            let (r_i_v, s_next) = sv.round(i, running, c0, c1, b"sumcheck/r");
            assert_eq!(r_i, r_i_v, "challenge mismatch at round {}", i);
            running = s_next;
        }

        let eval = sp.finalize_and_bind_eval();
        sv.finalize_and_check(eval, running);
    }

    #[test]
    fn e2e_sumcheck_merkle_folded_roundtrip() {
        let params = transcript::default_params();
        let p_tr = Transcript::new(b"SUMCHECK-MF-E2E", params.clone());
        let v_tr = Transcript::new(b"SUMCHECK-MF-E2E", params.clone());
        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        let ds_tag = F::from(6060u64);
        let merkle_cfg = MerkleChannelCfg::with_default_params(ds_tag);

        let mut rng = StdRng::seed_from_u64(1337);
        let k = 5usize;
        let n = 1usize << k;
        let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let mle = Mle::new(table.clone());

        let cfg = SumCheckMFConfig { queries_per_round: 3 };

        let mut sp = SumCheckMFProver::new(cfg, merkle_cfg.clone(), &mut pchan, &mle);

        let init_root = sp.cur.root;
        let mut sv =
            SumCheckMFVerifier::new(cfg, merkle_cfg.clone(), &mut vchan, init_root, k);
        sv.receive_initial_root(&init_root);

        let s = sp.send_claim();
        sv.recv_claim(&s);

        let mut s_running = s;
        let mut prev_root = init_root;
        let mut r_list: Vec<F> = Vec::with_capacity(k);

        for i in 0..k {
            let (c0, c1, r_i, next_root, openings) = sp.round(i);

            sv.start_round(i, s_running, c0, c1);

            let r_i_v = sv.derive_round_challenge(i);
            assert_eq!(r_i, r_i_v, "r_i mismatch at round {}", i);

            sv.recv_next_root(next_root);

            assert!(sv.verify_fold_openings(
                &openings.cur_indices,
                &openings.cur_values,
                &openings.cur_proof,
                &openings.next_indices,
                &openings.next_values,
                &openings.next_proof,
                r_i,
                prev_root,
                next_root
            ));

            s_running = sv.compute_s_next(c0, c1, r_i_v);

            prev_root = next_root;
            r_list.push(r_i);
        }

        let final_eval_prover = sp.finalize_eval();

        let mut offline = table.clone();
        for &rv in &r_list {
            let one_minus = F::from(1u64) - rv;
            for j in 0..(offline.len() / 2) {
                let a = offline[2 * j];
                let b = offline[2 * j + 1];
                offline[j] = one_minus * a + rv * b;
            }
            offline.truncate(offline.len() / 2);
        }
        assert_eq!(offline.len(), 1);
        let final_eval_offline = offline[0];

        assert_eq!(
            final_eval_offline, final_eval_prover,
            "offline f(r) != prover final eval"
        );

        sv.finalize_and_check(final_eval_prover, s_running);
    }
}