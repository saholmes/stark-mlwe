use ark_bls12_381::Fr as F;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

/// Perform IFFT in place on a slice by copying through a Vec.
/// If you can hold a Vec upstream, prefer the Vec version to avoid copies.
pub fn ifft_in_place(domain: &Radix2EvaluationDomain<F>, vals: &mut [F]) {
    let mut v: Vec<F> = vals.to_vec();
    domain.ifft_in_place(&mut v);
    vals.copy_from_slice(&v);
}

/// Perform FFT in place on a slice by copying through a Vec.
/// If you can hold a Vec upstream, prefer the Vec version to avoid copies.
pub fn fft_in_place(domain: &Radix2EvaluationDomain<F>, vals: &mut [F]) {
    let mut v: Vec<F> = vals.to_vec();
    domain.fft_in_place(&mut v);
    vals.copy_from_slice(&v);
}

/// Convenience helpers that allocate a new Vec and return results.

pub fn fft(domain: &Radix2EvaluationDomain<F>, coeffs: &[F]) -> Vec<F> {
    let mut v: Vec<F> = coeffs.to_vec();
    domain.fft_in_place(&mut v);
    v
}

pub fn ifft(domain: &Radix2EvaluationDomain<F>, evals: &[F]) -> Vec<F> {
    let mut v: Vec<F> = evals.to_vec();
    domain.ifft_in_place(&mut v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::One;

    #[test]
    fn roundtrip_fft_ifft() {
        let n = 8usize;
        let domain = Radix2EvaluationDomain::<F>::new(n).expect("domain");
        let mut coeffs = vec![F::one(); n];

        // slice-based in-place
        let evals = fft(&domain, &coeffs);
        let back = ifft(&domain, &evals);
        assert_eq!(coeffs, back);

        // mutate slice in-place via copy-through
        fft_in_place(&domain, &mut coeffs[..]);
        ifft_in_place(&domain, &mut coeffs[..]);
        assert_eq!(coeffs, vec![F::one(); n]);
    }
}
