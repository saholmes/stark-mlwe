pub use ark_bls12_381::Fr as F;
use ark_ff::{Field, PrimeField};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Domain {
    pub n: usize,
    pub omega: F,
    pub elements: Vec<F>,
}

/// Build a multiplicative subgroup H of size n (power of two) in Fr.
/// For BLS12-381 Fr, Radix-2 roots of unity are provided natively.
/// elements = [1, ω, ω^2, ..., ω^{n-1}], with ω^n = 1.
pub fn build_domain(n: usize) -> Domain {
    assert!(n.is_power_of_two(), "N must be power of two");
    // For power-of-two n, get the primitive 2^k-th root and set ω accordingly.
    let k = n.trailing_zeros() as usize;
    let omega = F::get_root_of_unity(k as u64).expect("root exists");

    let mut elements = Vec::with_capacity(n);
    let mut cur = F::from(1u64);
    for _ in 0..n {
        elements.push(cur);
        cur *= omega;
    }
    Domain { n, omega, elements }
}

/// Z_H(x) = x^n - 1 for H of size n.
pub fn zh_eval(x: F, n: usize) -> F {
    x.pow([n as u64]) - F::from(1u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_basic() {
        let n = 2048;
        let d = build_domain(n);
        assert_eq!(d.n, n);
        assert_eq!(d.elements.len(), n);
        // Check ω^n == 1
        assert_eq!(d.omega.pow([n as u64]), F::from(1u64));
        // Check vanishing polynomial over the domain
        for &h in &d.elements {
            assert_eq!(zh_eval(h, n), F::from(0u64));
        }
    }

    #[test]
    fn cyclic_property() {
        let n = 128;
        let d = build_domain(n);
        for i in 0..n {
            for j in 0..n {
                let a = d.elements[i] * d.elements[j];
                let b = d.elements[(i + j) % n];
                assert_eq!(a, b);
            }
        }
    }
}
