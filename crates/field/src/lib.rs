//! field crate: utilities around the BLS12-381 scalar field (Fr).
//!
//! Key points:
//! - No serde feature on ark-ff (0.5.x).
//! - Optional serde derives for your own types via the "serde1" feature.
//! - Prefer ark_serialize for canonical binary I/O of field elements.

pub use ark_pallas::Fr as F;
use ark_ff::{Field, FftField, One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Re-export the field type so users can depend on this crate for the field alias.
/// pub type F = Fr;

/// A simple multiplicative subgroup domain of size n = 2^log_n.
/// Stores the generator (omega) and optionally precomputes the elements.
///
/// Note:
/// - We do not derive Serialize/Deserialize unconditionally. If you enable the
///   "serde1" feature for this crate, derives will be enabled for Domain as well.
/// - Field elements should be serialized with ark_serialize for canonical formats.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Domain {
    /// Size of the domain (n)
    pub size: usize,
    /// log2(size)
    pub log_n: usize,
    /// A primitive n-th root of unity in the field
    pub omega: F,
    /// Optional cache of domain elements [1, omega, omega^2, ..., omega^{n-1}]
    pub elements: Vec<F>,
}

impl Domain {
    /// Construct a domain of size n = 2^log_n, returning None if such a root of unity
    /// does not exist in the field (should exist for Fr for reasonable sizes).
    pub fn new(log_n: usize) -> Option<Self> {
        let size = 1usize << log_n;
        // FftField::get_root_of_unity expects the size as a power-of-two exponent
        let omega = F::get_root_of_unity(size as u64)?;
        Some(Self {
            size,
            log_n,
            omega,
            elements: Vec::new(),
        })
    }

    /// Returns true if the domain has a non-zero size and a valid omega.
    pub fn is_valid(&self) -> bool {
        self.size > 0 && !self.omega.is_zero()
    }

    /// Returns (size, log_n).
    pub fn dims(&self) -> (usize, usize) {
        (self.size, self.log_n)
    }

    /// Precompute and store the domain elements in self.elements.
    /// elements = [1, omega, omega^2, ..., omega^{n-1}]
    pub fn precompute_elements(&mut self) {
        if self.size == 0 {
            self.elements.clear();
            return;
        }
        self.elements = compute_powers(self.omega, self.size);
    }

    /// Get a specific element omega^k. If elements are precomputed and k < size,
    /// return from the cache; otherwise compute on the fly.
    pub fn element(&self, k: usize) -> F {
        if let Some(v) = self.elements.get(k) {
            return *v;
        }
        self.omega.pow([k as u64])
    }

    /// Iterator over domain elements [1, omega, ..., omega^{n-1}].
    pub fn iter(&self) -> DomainIter {
        DomainIter {
            omega: self.omega,
            cur: F::one(),
            i: 0,
            n: self.size,
        }
    }
}

/// Iterator over [1, omega, omega^2, ..., omega^{n-1}]
pub struct DomainIter {
    omega: F,
    cur: F,
    i: usize,
    n: usize,
}

impl Iterator for DomainIter {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.n {
            return None;
        }
        let out = self.cur;
        self.cur *= self.omega;
        self.i += 1;
        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n.saturating_sub(self.i);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DomainIter {}

/// Compute [1, base, base^2, ..., base^{n-1}]
pub fn compute_powers(base: F, n: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(n);
    let mut cur = F::one();
    for _ in 0..n {
        v.push(cur);
        cur *= base;
    }
    v
}

/// Example helpers showing how to use ark_serialize for canonical I/O.
/// These serialize a single field element to bytes and back.

/// Serialize a field element to compressed bytes.
pub fn fr_to_bytes_compressed(x: &F) -> Vec<u8> {
    let mut out = Vec::new();
    x.serialize_with_mode(&mut out, Compress::Yes).expect("serialize");
    out
}

/// Deserialize a field element from compressed bytes, with validation.
pub fn fr_from_bytes_compressed(bytes: &[u8]) -> Result<F, ark_serialize::SerializationError> {
    F::deserialize_with_mode(bytes, Compress::Yes, Validate::Yes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_basic() {
        let log_n = 4; // size = 16
        let dom = Domain::new(log_n).expect("root of unity must exist");
        assert!(dom.is_valid());
        assert_eq!(dom.size, 16);
        assert_eq!(dom.dims(), (16, 4));

        // omega^n should be 1
        let w_n = dom.omega.pow([dom.size as u64]);
        assert!(w_n.is_one());
    }

    #[test]
    fn test_domain_iter_and_elements() {
        let mut dom = Domain::new(3).unwrap(); // size = 8
        dom.precompute_elements();

        let iter_elems: Vec<F> = dom.iter().collect();
        assert_eq!(iter_elems.len(), dom.size);
        assert_eq!(dom.elements.len(), dom.size);

        // spot check first few
        assert!(iter_elems[0].is_one());
        assert_eq!(iter_elems[1], dom.omega);
        assert_eq!(iter_elems[2], dom.omega * dom.omega);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let x = F::from(42u64);
        let bytes = fr_to_bytes_compressed(&x);
        let y = fr_from_bytes_compressed(&bytes).expect("deserialize");
        assert_eq!(x, y);
    }
}
