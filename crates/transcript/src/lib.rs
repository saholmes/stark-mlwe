use ark_ff::{PrimeField, Zero};
use ark_pallas::Fr as F;
use poseidon::{permute, PoseidonParams, RATE, T};

// Domain separation tags for transcript operations.
pub mod ds {
    pub const TRANSCRIPT_INIT: &[u8] = b"FSv1-TRANSCRIPT-INIT";
    pub const ABSORB_BYTES: &[u8] = b"FSv1-ABSORB-BYTES";
    pub const CHALLENGE: &[u8] = b"FSv1-CHALLENGE";
}

// Helper: map a byte string to a field element deterministically.
fn domain_tag_to_field(tag: &[u8]) -> F {
    // Interpret up to 32 bytes LE into field modulus.
    if tag.len() <= 32 {
        let mut le = [0u8; 32];
        le[..tag.len()].copy_from_slice(tag);
        F::from_le_bytes_mod_order(&le)
    } else {
        // Fold longer tags.
        let mut acc = F::zero();
        for chunk in tag.chunks(32) {
            let mut le = [0u8; 32];
            le[..chunk.len()].copy_from_slice(chunk);
            acc += F::from_le_bytes_mod_order(&le);
        }
        acc
    }
}

// Pack bytes into 31-byte field words (avoids reduction bias for Fr).
fn bytes_to_field_words(bytes: &[u8]) -> Vec<F> {
    const LIMB: usize = 31;
    let mut out = Vec::with_capacity((bytes.len() + LIMB - 1) / LIMB);
    for chunk in bytes.chunks(LIMB) {
        let mut le = [0u8; 32];
        le[..chunk.len()].copy_from_slice(chunk);
        out.push(F::from_le_bytes_mod_order(&le));
    }
    out
}

// Deterministic default parameters for transcript users.
pub fn default_params() -> PoseidonParams {
    poseidon::params::generate_params_t17_x5(b"POSEIDON-T17-X5-TRANSCRIPT")
}

pub struct Transcript {
    state: [F; T],
    pos: usize, // next rate lane to absorb into (0..RATE)
    params: PoseidonParams,
}

impl Transcript {
    pub fn new(label: &[u8], params: PoseidonParams) -> Self {
        let mut t = Transcript {
            state: [F::zero(); T],
            pos: 0,
            params,
        };
        // Initialize capacity with DS tag; absorb context label.
        t.state[T - 1] = domain_tag_to_field(ds::TRANSCRIPT_INIT);
        t.absorb_bytes(label);
        t
    }

    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        // Domain-separate the operation with a pre-absorb marker.
        self.absorb_field(domain_tag_to_field(ds::ABSORB_BYTES));
        // Break into fields and absorb each.
        let words = bytes_to_field_words(bytes);
        self.absorb_fields(&words);
    }

    pub fn absorb_field(&mut self, x: F) {
        self.absorb_fields(core::slice::from_ref(&x));
    }

    pub fn absorb_fields(&mut self, xs: &[F]) {
        for &x in xs {
            if self.pos == RATE {
                permute(&mut self.state, &self.params);
                self.pos = 0;
            }
            self.state[self.pos] += x;
            self.pos += 1;
        }
    }

    // Draw a challenge field element. We domain-separate by absorbing the label
    // and a CHALLENGE marker, then permute and output state[0].
    pub fn challenge(&mut self, label: &[u8]) -> F {
        self.absorb_field(domain_tag_to_field(ds::CHALLENGE));
        self.absorb_bytes(label);

        // Ensure we permute before reading
        permute(&mut self.state, &self.params);
        self.pos = 0; // reset rate cursor after permutation

        self.state[0]
    }

    pub fn challenges(&mut self, label: &[u8], n: usize) -> Vec<F> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut tag = Vec::with_capacity(label.len() + 8);
            tag.extend_from_slice(label);
            tag.extend_from_slice(&(i as u64).to_le_bytes());
            out.push(self.challenge(&tag));
        }
        out
    }

    pub fn params(&self) -> &PoseidonParams {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let params = default_params();

        let mut t1 = Transcript::new(b"ctx-A", params.clone());
        t1.absorb_bytes(b"hello");
        let c1 = t1.challenges(b"alpha", 3);

        let mut t2 = Transcript::new(b"ctx-A", params.clone());
        t2.absorb_bytes(b"hello");
        let c2 = t2.challenges(b"alpha", 3);

        assert_eq!(c1, c2);
    }

    #[test]
    fn sensitive_to_input() {
        let params = default_params();

        let mut t1 = Transcript::new(b"ctx-A", params.clone());
        t1.absorb_bytes(b"hello");
        let c1 = t1.challenge(b"alpha");

        let mut t2 = Transcript::new(b"ctx-A", params.clone());
        t2.absorb_bytes(b"hellp");
        let c2 = t2.challenge(b"alpha");

        assert_ne!(c1, c2);
    }
}
