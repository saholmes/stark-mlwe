use ark_ff::{Field, One, Zero};
use ark_pallas::Fr as F;

/// Return true if z ∈ H = <omega> of size n, i.e., z^n == 1.
fn is_in_domain(z: F, n: usize) -> bool {
    z.pow(&[n as u64, 0, 0, 0]) == F::one()
}

/// Vanishing polynomial on H: Z_H(z) = z^n - 1
fn zh_at(z: F, n: usize) -> F {
    z.pow(&[n as u64, 0, 0, 0]) - F::one()
}

/// Evaluate the unique degree < n polynomial with values = {f(ω^j)} on H at z ∉ H:
/// Using the correct barycentric form for multiplicative subgroup H:
/// f(z) = (Z_H(z)/n) * sum_j f(ω^j) * ω^j / (z - ω^j)
pub fn lagrange_eval_on_h(values: &[F], z: F, omega: F) -> F {
    let n = values.len();
    assert!(n > 0, "non-empty domain");
    if is_in_domain(z, n) {
        // Exact evaluation on-grid by lookup
        let mut omega_j = F::one();
        for j in 0..n {
            if z == omega_j {
                return values[j];
            }
            omega_j *= omega;
        }
        panic!("z in domain but not matching a power of omega");
    }

    let zh = zh_at(z, n);
    let n_inv = F::from(n as u64)
        .inverse()
        .expect("n invertible in prime field");

    let mut sum = F::zero();
    let mut omega_j = F::one(); // ω^0
    for j in 0..n {
        let inv = (z - omega_j).inverse().expect("z ∉ H");
        sum += values[j] * omega_j * inv;
        omega_j *= omega;
    }
    zh * n_inv * sum
}

/// DEEP-ALI merge (no blinding).
/// Input evaluations A,S,E,T over H, primitive root omega, and challenge z ∉ H.
/// Returns:
/// - f0 evaluations on H where f0(x) = (Φ(x) − c*·Z_H(x)) / (x − z),
/// - z,
/// - c* = Φ(z)/Z_H(z),
/// with Φ = A·S + E − T.
pub fn deep_ali_merge_evals(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    omega: F,
    z: F,
) -> (Vec<F>, F, F) {
    let n = a_eval.len();
    assert!(n > 1);
    assert_eq!(s_eval.len(), n);
    assert_eq!(e_eval.len(), n);
    assert_eq!(t_eval.len(), n);
    assert!(!is_in_domain(z, n), "z must be outside H");

    // Φ on H
    let mut phi_eval = vec![F::zero(); n];
    for i in 0..n {
        phi_eval[i] = a_eval[i] * s_eval[i] + e_eval[i] - t_eval[i];
    }

    // Φ(z) via Lagrange
    let phi_z = lagrange_eval_on_h(&phi_eval, z, omega);
    let zh_z = zh_at(z, n);
    let c_star = phi_z * zh_z.inverse().expect("z ∉ H ⇒ Z_H(z) ≠ 0");

    // f0 on H: Z_H(ω^j)=0 ⇒ f0(ω^j) = Φ(ω^j)/(ω^j − z)
    let mut f0_eval = vec![F::zero(); n];
    let mut omega_j = F::one();
    for j in 0..n {
        f0_eval[j] = phi_eval[j] * (omega_j - z).inverse().expect("z ∉ H");
        omega_j *= omega;
    }

    (f0_eval, z, c_star)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_poly::EvaluationDomain;
    use rand::{rngs::StdRng, SeedableRng};

    /// Horner evaluation of a polynomial with coefficients coeffs.
    fn poly_eval(coeffs: &[F], x: F) -> F {
        let mut acc = F::zero();
        for &c in coeffs.iter().rev() {
            acc *= x;
            acc += c;
        }
        acc
    }

    /// Evaluate coeffs on H = <omega> of size n (deg < n).
    fn eval_on_domain(coeffs: &[F], omega: F, n: usize) -> Vec<F> {
        let mut out = Vec::with_capacity(n);
        let mut x = F::one();
        for _ in 0..n {
            out.push(poly_eval(coeffs, x));
            x *= omega;
        }
        out
    }

    /// Find a primitive n-th root of unity for power-of-two n via ark-poly Radix2 domain.
    fn find_primitive_root_pow2(n: usize) -> F {
        use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
        let dom = Domain::<F>::new(n).expect("radix-2 domain exists for this n");
        dom.group_gen
    }

    #[test]
    fn test_lagrange_eval_matches_polyeval_off_domain() {
        let n = 32usize;
        let omega = find_primitive_root_pow2(n);
        // Sample a random polynomial of degree < n
        let mut rng = StdRng::seed_from_u64(7);
        let deg = 20usize;
        let coeffs: Vec<F> = (0..=deg).map(|_| F::rand(&mut rng)).collect();
        let evals = eval_on_domain(&coeffs, omega, n);

        // Pick random z ∉ H
        let mut rngz = StdRng::seed_from_u64(99);
        let z = loop {
            let z = F::rand(&mut rngz);
            if !is_in_domain(z, n) {
                break z;
            }
        };

        let fz_eval = lagrange_eval_on_h(&evals, z, omega);
        let fz_poly = poly_eval(&coeffs, z);
        assert_eq!(fz_eval, fz_poly, "lagrange evaluation must match direct poly eval");
    }

    #[test]
    fn test_deep_ali_merge_completeness() {
        let n = 64usize;
        let omega = find_primitive_root_pow2(n);

        let mut rng = StdRng::seed_from_u64(2024);
        // Degrees strictly less than n
        let deg_a = 15usize;
        let deg_s = 17usize;
        let deg_e = 7usize;
        let a_coeffs: Vec<F> = (0..=deg_a).map(|_| F::rand(&mut rng)).collect();
        let s_coeffs: Vec<F> = (0..=deg_s).map(|_| F::rand(&mut rng)).collect();
        let e_coeffs: Vec<F> = (0..=deg_e).map(|_| F::rand(&mut rng)).collect();

        // Evaluate on H
        let a_eval = eval_on_domain(&a_coeffs, omega, n);
        let s_eval = eval_on_domain(&s_coeffs, omega, n);
        let e_eval = eval_on_domain(&e_coeffs, omega, n);

        // Compute T = A*S + E on H
        let mut t_eval = vec![F::zero(); n];
        for i in 0..n {
            t_eval[i] = a_eval[i] * s_eval[i] + e_eval[i];
        }

        // Pick z ∉ H
        let mut rngz = StdRng::seed_from_u64(8080);
        let z = loop {
            let cand = F::rand(&mut rngz);
            if !is_in_domain(cand, n) {
                break cand;
            }
        };

        let (f0_eval, z_out, c_star) =
            deep_ali_merge_evals(&a_eval, &s_eval, &e_eval, &t_eval, omega, z);
        assert_eq!(z_out, z);

        // Spot-check identity at a few grid points x ∈ H:
        for &j in &[0usize, 1, 7, 13, 31, 47, 63] {
            let x = omega.pow(&[j as u64, 0, 0, 0]);
            let phi_x = a_eval[j] * s_eval[j] + e_eval[j] - t_eval[j];
            let zh_x = zh_at(x, n);
            let left = f0_eval[j] * (x - z) + c_star * zh_x;
            assert_eq!(left, phi_x, "merged identity must hold at grid point {j}");
        }
    }
}