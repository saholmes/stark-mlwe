use ark_serialize::CanonicalSerialize;
use ark_ff::{Field, One, Zero};
use ark_pallas::Fr as F;
use rand::{rngs::StdRng, Rng, SeedableRng};


/// FRI multiplicative domain descriptor.
#[derive(Clone, Copy, Debug)]
pub struct FriDomain {
    pub omega: F,
    pub size: usize,
}

impl FriDomain {
    pub fn new_radix2(size: usize) -> Self {
        use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
        use ark_poly::EvaluationDomain;
        let dom = Domain::<F>::new(size).expect("radix-2 domain exists");
        Self {
            omega: dom.group_gen,
            size,
        }
    }
}

/// Placeholder for Poseidon parameters; not used in this milestone.
#[derive(Clone, Debug, Default)]
pub struct PoseidonParamsPlaceholder;

/// FRI layer configuration.
#[derive(Clone, Debug)]
pub struct FriLayerParams {
    pub m: usize,                  // folding factor
    pub width: usize,              // arity/coset width; equals m here
    pub poseidon_params: PoseidonParamsPlaceholder,
    pub level: usize,              // layer index ℓ
}

impl FriLayerParams {
    pub fn new(level: usize, m: usize) -> Self {
        Self {
            m,
            width: m,
            poseidon_params: PoseidonParamsPlaceholder::default(),
            level,
        }
    }
}

/// Deterministic per-layer challenge sampler based on a seed and "FRI/z/l" tag.
/// In production, derive from a transcript; here we use a reproducible RNG.
pub fn fri_sample_z_ell(seed: u64, level: usize, domain_size: usize) -> F {
    let mut rng = StdRng::seed_from_u64(seed ^ (0xF1F1_0000_0000_0000u64 + level as u64));
    loop {
        let cand = F::from(rng.gen::<u64>());
        // Require z not in the current domain (size at this level).
        if cand.pow(&[domain_size as u64, 0, 0, 0]) != F::one() {
            return cand;
        }
    }
}

/// Fold one FRI layer.
/// Input: f_l over H_l (size N), folding factor m, challenge z_l.
/// Output: f_{l+1} over reduced domain H_{l+1} of size N/m.
///
/// For this milestone, we define the fold as:
/// - Partition f_l into chunks of size m: [v_0, ..., v_{m-1}] per bucket.
/// - Output value is a simple linear combination with powers of z_l:
///     g = sum_{i=0}^{m-1} v_i * z_l^i
/// This is only a plumbing placeholder; real FRI uses a specific structured fold.
pub fn fri_fold_layer(f_l: &[F], z_l: F, m: usize) -> Vec<F> {
    assert!(m >= 2);
    assert!(f_l.len() % m == 0, "layer size must be divisible by m");
    let n_next = f_l.len() / m;
    let mut out = vec![F::zero(); n_next];

    // Precompute powers of z_l up to m-1
    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m {
        z_pows.push(acc);
        acc *= z_l;
    }

    for b in 0..n_next {
        let base = b * m;
        let mut s = F::zero();
        for i in 0..m {
            s += f_l[base + i] * z_pows[i];
        }
        out[b] = s;
    }
    out
}

/// Fold across a whole schedule, returning all layer vectors [f_0, f_1, ..., f_L].
pub fn fri_fold_schedule(f0: Vec<F>, schedule: &[usize], seed: u64) -> Vec<Vec<F>> {
    let mut layers = Vec::with_capacity(schedule.len() + 1);
    let mut cur = f0;
    let mut cur_size = cur.len();
    layers.push(cur.clone());

    for (level, &m) in schedule.iter().enumerate() {
        assert!(cur_size % m == 0, "size must be divisible by m at level {level}");
        let z_l = fri_sample_z_ell(seed, level, cur_size);
        let next = fri_fold_layer(&cur, z_l, m);
        cur = next;
        cur_size = cur.len();
        layers.push(cur.clone());
    }
    layers
}

/// A trivial commitment for testing: hash each field element with blake3.
#[derive(Clone, Debug)]
pub struct FriCommitment {
    pub digests: Vec<[u8; 32]>,
}

impl FriCommitment {
    pub fn commit(values: &[F]) -> Self {
        let mut digests = Vec::with_capacity(values.len());
        for v in values {
            let mut buf = [0u8; 32];
            v.serialize_uncompressed(&mut buf[..]).expect("serialize");
            let hash = blake3::hash(&buf);
            digests.push(*hash.as_bytes());
        }
        Self { digests }
    }

    pub fn open(&self, idx: usize) -> [u8; 32] {
        self.digests[idx]
    }

    pub fn verify(&self, idx: usize, v: F) -> bool {
        let mut buf = [0u8; 32];
        v.serialize_uncompressed(&mut buf[..]).expect("serialize");
        blake3::hash(&buf).as_bytes() == &self.digests[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;

    #[test]
    fn test_fri_fold_shape() {
        // Start with N0 divisible by 16*16*8 = 2048
        let n0 = 4096usize;
        let schedule = [16usize, 16usize, 8usize];
        let mut rng = StdRng::seed_from_u64(7777);

        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();

        let layers = fri_fold_schedule(f0.clone(), &schedule, 0xDEADBEEF);
        assert_eq!(layers.len(), schedule.len() + 1);

        let n1 = n0 / 16;
        let n2 = n1 / 16;
        let n3 = n2 / 8;
        assert_eq!(layers[0].len(), n0);
        assert_eq!(layers[1].len(), n1);
        assert_eq!(layers[2].len(), n2);
        assert_eq!(layers[3].len(), n3);
    }

   #[test]
    fn test_layer_commit_open_basic() {
        let n0 = 4096usize; // divisible by 16*16*8 = 2048
        let schedule = [16usize, 16usize, 8usize];
        let mut rng = StdRng::seed_from_u64(8888);

        let f0: Vec<F> = (0..n0).map(|_| F::rand(&mut rng)).collect();
        let layers = fri_fold_schedule(f0.clone(), &schedule, 0xBAD5EED);

        // For each layer, commit and verify a few random indices.
        for (ell, f) in layers.iter().enumerate() {
            let com = FriCommitment::commit(f);

            // probe 5 random positions
            let mut rngp = StdRng::seed_from_u64(ell as u64 * 101);
            for _ in 0..5 {
                let idx = (rngp.gen::<usize>() % f.len()).min(f.len().saturating_sub(1));
                assert!(com.verify(idx, f[idx]));
                let opened = com.open(idx);
                assert_eq!(opened, com.digests[idx]);
            }
        }
    }
}